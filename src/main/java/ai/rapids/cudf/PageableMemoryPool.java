/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * JNI interface to a rmm::pool_memory_resource backed by a host memory resource that
 * pre-touches each allocated page in parallel. Intended as a fallback when the pinned
 * pool is exhausted: a pre-touched pageable destination reaches a much higher fraction
 * of pinned DtoH bandwidth than freshly-malloc'd pageable, because the page faults are
 * paid at pool growth time rather than at DtoH copy time.
 */
public final class PageableMemoryPool implements AutoCloseable {
  private static final Logger log = LoggerFactory.getLogger(PageableMemoryPool.class);
  private static final boolean TRACE_STAGED_COPY_CHUNKS = Boolean.getBoolean(
      "ai.rapids.cudf.pageable.staged.copy.trace.chunks");

  // These static fields should only ever be accessed when class-synchronized.
  // Do NOT use singleton_ directly!  Use the getSingleton accessor instead.
  private static volatile PageableMemoryPool singleton_ = null;
  private static volatile Future<PageableMemoryPool> initFuture = null;
  private final ReentrantReadWriteLock lifecycleLock = new ReentrantReadWriteLock();
  private static ExecutorService stagedCopyExecutor = null;
  // A copy owns its CUDA event and staging buffers until all worker futures complete.
  // Fairness prevents new callers from overtaking shutdown once it is waiting.
  private static final ReentrantReadWriteLock stagedCopyLifecycleLock =
      new ReentrantReadWriteLock(true);
  private static ScheduledExecutorService debugPoolStateExecutor = null;
  private static HostMemoryBuffer[] stagedCopyBuffers = null;
  private static long[] registeredStagedCopyBuffers = null;
  private static volatile boolean asyncStagedD2H = false;
  private static ArrayBlockingQueue<AsyncD2HSlot> freeAsyncD2HSlots = null;
  private static BlockingQueue<PendingAsyncD2H> pendingAsyncD2H = null;
  private static List<AsyncD2HSlot> asyncD2HSlots = null;
  private static Thread asyncD2HPoller = null;
  private static volatile boolean stopAsyncD2HPoller = false;
  private static int stagedCopyThreads = 0;
  private static int asyncD2HSlotCount = 0;
  private static long stagedCopyChunkSize = 0;
  private static long stagedCopyMinSize = 0;
  private static volatile Throwable asyncD2HFailure = null;
  private static final AtomicLong stagedCopyCallCount = new AtomicLong();
  private static final AtomicLong stagedD2HCalls = new AtomicLong();
  private static final AtomicLong stagedD2HBytes = new AtomicLong();
  private static final AtomicLong stagedH2DCalls = new AtomicLong();
  private static final AtomicLong stagedH2DBytes = new AtomicLong();
  private static final AtomicLong stagedCopySkippedSmallCount = new AtomicLong();
  private static final AtomicLong stagedCopySkippedDisabledCount = new AtomicLong();
  private static final AtomicLong stagedCopySkippedNonPoolCount = new AtomicLong();
  private static final AtomicLong stagedCopySkippedNoExecutorCount = new AtomicLong();
  private static final AtomicLong stagedCopyBytes = new AtomicLong();
  private static final AtomicLong stagedCopyChunks = new AtomicLong();
  private static final AtomicLong stagedCopyTotalNs = new AtomicLong();
  private static final AtomicLong stagedCopyRecordEventNs = new AtomicLong();
  private static final AtomicLong stagedCopySubmitNs = new AtomicLong();
  private static final AtomicLong stagedCopyWaitNs = new AtomicLong();
  private static final AtomicLong stagedCopyQueueNs = new AtomicLong();
  private static final AtomicLong stagedCopyMaxQueueNs = new AtomicLong();
  private static final AtomicLong stagedCopyWorkerActive = new AtomicLong();
  private static final AtomicLong stagedCopyMaxWorkerActive = new AtomicLong();
  private static final AtomicLong stagedCopyMaxBytes = new AtomicLong();
  private static final AtomicLong stagedCopyInFlight = new AtomicLong();
  private static final AtomicLong stagedCopyMaxInFlight = new AtomicLong();
  private static final AtomicLong asyncD2HSlotWaitNs = new AtomicLong();
  private static final AtomicLong asyncD2HMaxSlotWaitNs = new AtomicLong();
  private static final AtomicLong asyncD2HCompletionPolls = new AtomicLong();
  private static final AtomicLong poolAllocCalls = new AtomicLong();
  private static final AtomicLong poolAllocSuccess = new AtomicLong();
  private static final AtomicLong poolAllocMiss = new AtomicLong();
  private static final AtomicLong poolAllocBytes = new AtomicLong();
  private static final AtomicLong poolAllocMissBytes = new AtomicLong();
  private static final AtomicLong maxPoolAllocBytes = new AtomicLong();
  private static final AtomicLong maxPoolAllocMissBytes = new AtomicLong();
  private static final AtomicLong poolLiveBytes = new AtomicLong();
  private static final AtomicLong poolMaxLiveBytes = new AtomicLong();
  private long poolHandle;
  private long poolSize;

  private static final long DEFAULT_STAGED_COPY_CHUNK_SIZE = 64L * 1024L * 1024L;

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static native long newPageablePoolMemoryResource(long poolSize, int pretouchThreads,
                                                           boolean numaBind, boolean thpDeny);

  private static native void releasePageablePoolMemoryResource(long poolPtr);

  private static native long allocFromPageablePool(long poolPtr, long size);

  private static native void freeFromPageablePool(long poolPtr, long ptr, long size);

  private static native void stagedCopyFromDevice(long dst, long src, long size, long waitEvent);

  private static native void stagedCopyToDevice(long dst, long src, long size, long waitEvent);

  private static native void stagedHostCopy(long dst, long src, long size);

  private static native void setStagingBuffer(long address, long size);

  private static native long allocateRegisteredStagingBuffer(long size);

  private static native void freeRegisteredStagingBuffer(long address);

  private static native String getAndResetStagedCopyNativeStats();


  private static final class AsyncD2HSlot implements AutoCloseable {
    final long address;
    final Cuda.Stream stream;
    final Cuda.Event done;

    AsyncD2HSlot(long address) {
      this.address = address;
      Cuda.Stream newStream = null;
      try {
        newStream = new Cuda.Stream(true);
        done = new Cuda.Event(false, false);
        stream = newStream;
      } catch (RuntimeException | Error e) {
        if (newStream != null) {
          newStream.close();
        }
        freeRegisteredStagingBuffer(address);
        throw e;
      }
    }

    @Override
    public void close() {
      done.close();
      stream.close();
      freeRegisteredStagingBuffer(address);
    }
  }

  private static final class PendingAsyncD2H {
    final AsyncD2HSlot slot;
    final long destination;
    final long size;
    final CompletableFuture<Void> future;

    PendingAsyncD2H(AsyncD2HSlot slot, long destination, long size,
        CompletableFuture<Void> future) {
      this.slot = slot;
      this.destination = destination;
      this.size = size;
      this.future = future;
    }
  }

  private static final class PageableHostBufferCleaner extends MemoryBuffer.MemoryBufferCleaner {
    private long address;
    private final long origLength;
    private final boolean stageDeviceToHost;

    PageableHostBufferCleaner(long address, long length, boolean stageDeviceToHost) {
      this.address = address;
      this.origLength = length;
      this.stageDeviceToHost = stageDeviceToHost;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long origAddress = 0;
      if (address != -1) {
        origAddress = address;
        try {
          PageableMemoryPool.freeInternal(address, origLength);
        } finally {
          poolLiveBytes.addAndGet(-origLength);
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          address = -1;
        }
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A PAGEABLE HOST BUFFER WAS LEAKED (ID: " + id + " "
            + Long.toHexString(origAddress) + ")");
        logRefCountDebug("Leaked pageable host buffer");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return address == -1;
    }
  }

  /** Owns a raw allocation while marking it for the same D2H staging path as pool buffers. */
  private static final class RawStagedHostBufferCleaner extends MemoryBuffer.MemoryBufferCleaner {
    private HostMemoryBuffer raw;

    RawStagedHostBufferCleaner(HostMemoryBuffer raw) {
      this.raw = raw;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      if (raw == null) {
        return false;
      }
      try {
        if (logErrorIfNotClean) {
          log.error("A RAW STAGED HOST BUFFER WAS LEAKED (ID: " + id + ")");
          logRefCountDebug("Leaked raw staged host buffer");
        }
        raw.close();
      } finally {
        raw = null;
      }
      return true;
    }

    @Override
    public boolean isClean() {
      return raw == null;
    }
  }

  /** Allocate outside the pool, retaining explicit D2H staging for allocator-based callers. */
  public static HostMemoryBuffer allocateRawForDeviceToHost(long bytes) {
    HostMemoryBuffer raw = HostMemoryBuffer.allocateRaw(bytes);
    try {
      return new HostMemoryBuffer(raw.getAddress(), bytes, new RawStagedHostBufferCleaner(raw));
    } catch (Throwable t) {
      raw.close();
      throw t;
    }
  }

  private static PageableMemoryPool getSingleton() {
    if (singleton_ == null) {
      if (initFuture == null) {
        return null;
      }
      synchronized (PageableMemoryPool.class) {
        if (singleton_ == null && initFuture != null) {
          try {
            singleton_ = initFuture.get();
          } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted waiting for pageable pool initialization", e);
          } catch (Exception e) {
            initFuture = null;
            throw new RuntimeException("Error initializing pageable memory pool", e);
          }
          initFuture = null;
        }
      }
    }
    return singleton_;
  }

  private static void freeInternal(long address, long origLength) {
    PageableMemoryPool pool = getSingleton();
    if (pool == null) {
      throw new IllegalStateException("Cannot free pageable buffer after pool shutdown");
    }
    pool.free(address, origLength);
  }

  /**
   * Initialize the pool. The backing buffer is allocated and pre-touched (each system page
   * written to force the kernel to map physical pages) using the given thread count.
   * Pre-touching parallelizes the page-fault work so it amortizes to a few hundred ms
   * for multi-GB pools instead of several seconds serially.
   *
   * @param poolSize        size of the pool in bytes
   * @param pretouchThreads number of worker threads to use for the parallel pre-touch
   */
  public static synchronized void initialize(long poolSize, int pretouchThreads) {
    initialize(poolSize, pretouchThreads, false);
  }

  public static synchronized void initialize(long poolSize, int pretouchThreads,
      boolean numaBind) {
    if (isInitialized()) {
      throw new IllegalStateException("Can only initialize the pageable pool once.");
    }
    int copyThreads = Integer.getInteger("ai.rapids.cudf.pageable.staged.copy.threads", 8);
    stagedCopyThreads = Math.max(1, copyThreads);
    stagedCopyChunkSize = Long.getLong("ai.rapids.cudf.pageable.staged.copy.chunk.bytes",
        DEFAULT_STAGED_COPY_CHUNK_SIZE);
    stagedCopyChunkSize = Math.max(1L, stagedCopyChunkSize);
    stagedCopyMinSize = Long.getLong("ai.rapids.cudf.pageable.staged.copy.min.bytes", 0L);
    stagedCopyMinSize = Math.max(0L, stagedCopyMinSize);
    asyncStagedD2H = Boolean.getBoolean("ai.rapids.cudf.pageable.staged.copy.async.d2h");
    asyncD2HSlotCount = Math.max(1, Integer.getInteger(
        "ai.rapids.cudf.pageable.staged.copy.async.d2h.slots", stagedCopyThreads));
    boolean registeredBuffers = Boolean.getBoolean(
        "ai.rapids.cudf.pageable.staged.copy.registered.buffers");
    // The pageable allocator is also used without a pinned pool (including its own
    // allocation tests). In that case it can still initialize, but staged copies
    // are unavailable until the next initialization with a pinned pool.
    if (registeredBuffers || PinnedMemoryPool.isInitialized()) {
    AtomicInteger stagedCopyThreadId = new AtomicInteger();
    stagedCopyExecutor = Executors.newFixedThreadPool(stagedCopyThreads, new ThreadFactory() {
      @Override
      public Thread newThread(Runnable runnable) {
        Thread t = new Thread(runnable,
            "pageable staged copy " + stagedCopyThreadId.getAndIncrement());
        t.setDaemon(true);
        return t;
      }
    });
    stagedCopyBuffers = new HostMemoryBuffer[stagedCopyThreads];
    registeredStagedCopyBuffers = new long[stagedCopyThreads];
    List<Future<?>> warmupFutures = new ArrayList<>(stagedCopyThreads);
    try {
      for (int i = 0; i < stagedCopyThreads; i++) {
        long address;
        if (registeredBuffers) {
          address = allocateRegisteredStagingBuffer(stagedCopyChunkSize);
          registeredStagedCopyBuffers[i] = address;
        } else {
          HostMemoryBuffer buffer = PinnedMemoryPool.tryAllocate(stagedCopyChunkSize);
          if (buffer == null) {
            throw new IllegalStateException("Unable to allocate " + stagedCopyChunkSize
                + " byte pinned staging buffer " + (i + 1) + " of " + stagedCopyThreads
                + " from the pinned pool. Increase the pinned pool size.");
          }
          stagedCopyBuffers[i] = buffer;
          address = buffer.getAddress();
        }
        warmupFutures.add(stagedCopyExecutor.submit(() ->
            setStagingBuffer(address, stagedCopyChunkSize)));
      }
      waitForStagedCopies(warmupFutures);
      if (asyncStagedD2H) {
        initializeAsyncD2HSlots();
      }
    } catch (RuntimeException e) {
      stopAsyncD2HSlots();
      stagedCopyExecutor.shutdownNow();
      stagedCopyExecutor = null;
      closeStagedCopyBuffers();
      throw e;
    }
    }
    long poolStateEveryMs = Long.getLong(
        "ai.rapids.cudf.pageable.pool.debug.state.every.ms", 0L);
    if (poolStateEveryMs > 0) {
      debugPoolStateExecutor = Executors.newSingleThreadScheduledExecutor(runnable -> {
        Thread t = new Thread(runnable, "pageable pool debug state");
        t.setDaemon(true);
        return t;
      });
      debugPoolStateExecutor.scheduleAtFixedRate(PageableMemoryPool::logPoolState,
          poolStateEveryMs, poolStateEveryMs, TimeUnit.MILLISECONDS);
    }
    boolean thpDeny = "deny".equals(System.getenv("SPARK_RAPIDS_PAGEABLE_POOL_THP"));
    FutureTask<PageableMemoryPool> task = new FutureTask<>(
        () -> new PageableMemoryPool(poolSize, pretouchThreads, numaBind, thpDeny));
    Thread initThread = new Thread(task, "pageable pool init");
    initThread.setDaemon(true);
    initThread.start();
    initFuture = task;
  }

  /**
   * Check if the pool has been initialized or not.
   */
  public static boolean isInitialized() {
    return getSingleton() != null;
  }

  /** Whether plugin D2H routes should use the staged copy path. */
  public static boolean isDeviceToHostStagingEnabled() {
    return !Boolean.getBoolean("ai.rapids.cudf.pageable.staged.copy.d2h.disabled") &&
        isInitialized();
  }

  /**
   * Shut down the pool, nulling out our reference. Any allocation or free in flight
   * will fail after this.
   */
  public static synchronized void shutdown() {
    stagedCopyLifecycleLock.writeLock().lock();
    try {
      PageableMemoryPool pool = getSingleton();
      stopAsyncD2HSlots();
      if (stagedCopyExecutor != null) {
        stagedCopyExecutor.shutdown();
        stagedCopyExecutor = null;
      }
      if (debugPoolStateExecutor != null) {
        debugPoolStateExecutor.shutdown();
        debugPoolStateExecutor = null;
      }
      closeStagedCopyBuffers();
      logPoolState();
      logStagedCopyStats();
      if (pool != null) {
        pool.close();
      }
      initFuture = null;
      singleton_ = null;
      asyncStagedD2H = false;
    } finally {
      stagedCopyLifecycleLock.writeLock().unlock();
    }
  }

  private static void initializeAsyncD2HSlots() {
    ArrayBlockingQueue<AsyncD2HSlot> free = new ArrayBlockingQueue<>(asyncD2HSlotCount);
    List<AsyncD2HSlot> slots = new ArrayList<>(asyncD2HSlotCount);
    try {
      for (int i = 0; i < asyncD2HSlotCount; i++) {
        long address = allocateRegisteredStagingBuffer(stagedCopyChunkSize);
        AsyncD2HSlot slot = new AsyncD2HSlot(address);
        slots.add(slot);
        free.add(slot);
      }
    } catch (RuntimeException e) {
      for (AsyncD2HSlot slot : slots) {
        slot.close();
      }
      throw e;
    }
    freeAsyncD2HSlots = free;
    pendingAsyncD2H = new LinkedBlockingQueue<>();
    asyncD2HSlots = slots;
    asyncD2HFailure = null;
    stopAsyncD2HPoller = false;
    asyncD2HPoller = new Thread(PageableMemoryPool::pollAsyncD2H, "pageable D2H event poller");
    asyncD2HPoller.setDaemon(true);
    asyncD2HPoller.start();
  }

  private static void stopAsyncD2HSlots() {
    Thread poller = asyncD2HPoller;
    if (poller != null) {
      stopAsyncD2HPoller = true;
      LockSupport.unpark(poller);
      try {
        poller.join();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new RuntimeException("Interrupted while stopping staged D2H poller", e);
      }
      asyncD2HPoller = null;
    }
    List<AsyncD2HSlot> slots = asyncD2HSlots;
    asyncD2HSlots = null;
    freeAsyncD2HSlots = null;
    pendingAsyncD2H = null;
    if (slots != null) {
      for (AsyncD2HSlot slot : slots) {
        slot.close();
      }
    }
  }

  private static void pollAsyncD2H() {
    List<PendingAsyncD2H> active = new ArrayList<>(asyncD2HSlotCount);
    BlockingQueue<PendingAsyncD2H> pending = pendingAsyncD2H;
    while (!stopAsyncD2HPoller || !active.isEmpty() || !pending.isEmpty()) {
      pending.drainTo(active);
      boolean completed = false;
      for (int i = active.size() - 1; i >= 0; i--) {
        PendingAsyncD2H copy = active.get(i);
        asyncD2HCompletionPolls.incrementAndGet();
        boolean ready;
        try {
          ready = copy.slot.done.hasCompleted();
        } catch (RuntimeException | Error t) {
          asyncD2HFailure = t;
          copy.future.completeExceptionally(t);
          active.remove(i);
          completed = true;
          continue;
        }
        if (ready) {
          active.remove(i);
          completed = true;
          long queuedAt = System.nanoTime();
          try {
            stagedCopyExecutor.execute(() -> {
              long queueNs = System.nanoTime() - queuedAt;
              stagedCopyQueueNs.addAndGet(queueNs);
              updateMax(stagedCopyMaxQueueNs, queueNs);
              long activeWorkers = stagedCopyWorkerActive.incrementAndGet();
              updateMax(stagedCopyMaxWorkerActive, activeWorkers);
              Throwable failure = null;
              try (NvtxRange range = new NvtxRange("asyncStagedD2HHostCopy", NvtxColor.YELLOW)) {
                stagedHostCopy(copy.destination, copy.slot.address, copy.size);
              } catch (Throwable t) {
                failure = t;
              } finally {
                freeAsyncD2HSlots.add(copy.slot);
                if (failure == null) {
                  copy.future.complete(null);
                } else {
                  copy.future.completeExceptionally(failure);
                }
                stagedCopyWorkerActive.decrementAndGet();
              }
            });
          } catch (RuntimeException | Error t) {
            asyncD2HFailure = t;
            copy.future.completeExceptionally(t);
          }
        }
      }
      if (!completed) {
        LockSupport.parkNanos(active.isEmpty() ? 100_000L : 20_000L);
      }
    }
  }

  private static void copyFromDeviceAsync(long dst, long src, long length, Cuda.Event ready) {
    ArrayBlockingQueue<AsyncD2HSlot> free = freeAsyncD2HSlots;
    BlockingQueue<PendingAsyncD2H> pending = pendingAsyncD2H;
    List<Future<?>> futures = new ArrayList<>();
    long submitStart = System.nanoTime();
    long offset = 0;
    try {
      while (offset < length) {
        long chunk = Math.min(stagedCopyChunkSize, length - offset);
        long slotWaitStart = System.nanoTime();
        AsyncD2HSlot slot;
        try {
          do {
            if (asyncD2HFailure != null) {
              throw new IllegalStateException("Asynchronous staged D2H poller failed",
                  asyncD2HFailure);
            }
            slot = free.poll(100, TimeUnit.MILLISECONDS);
          } while (slot == null);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException("Interrupted while waiting for a D2H staging buffer", e);
        }
        long slotWait = System.nanoTime() - slotWaitStart;
        asyncD2HSlotWaitNs.addAndGet(slotWait);
        updateMax(asyncD2HMaxSlotWaitNs, slotWait);
        CompletableFuture<Void> future = new CompletableFuture<>();
        try {
          slot.stream.waitOn(ready);
          Cuda.asyncMemcpy(slot.address, src + offset, chunk,
              CudaMemcpyKind.DEVICE_TO_HOST, slot.stream);
          slot.done.record(slot.stream);
          pending.add(new PendingAsyncD2H(slot, dst + offset, chunk, future));
          LockSupport.unpark(asyncD2HPoller);
          futures.add(future);
        } catch (RuntimeException | Error t) {
          try {
            slot.stream.sync();
          } finally {
            free.add(slot);
          }
          throw t;
        }
        offset += chunk;
        stagedCopyChunks.incrementAndGet();
      }
    } finally {
      stagedCopySubmitNs.addAndGet(System.nanoTime() - submitStart);
      long waitStart = System.nanoTime();
      waitForStagedCopies(futures);
      stagedCopyWaitNs.addAndGet(System.nanoTime() - waitStart);
    }
  }

  /**
   * Factory method to create a pageable host memory buffer.
   *
   * @param bytes size in bytes to allocate
   * @return newly created buffer, or null if the pool is uninitialized or exhausted
   *         (caller should fall back to a regular malloc'd buffer)
   */
  public static HostMemoryBuffer tryAllocate(long bytes) {
    return tryAllocate(bytes, false);
  }

  /** Allocate a pageable destination whose device-to-host copies must use pinned staging. */
  public static HostMemoryBuffer tryAllocateForDeviceToHost(long bytes) {
    return tryAllocate(bytes, true);
  }

  private static HostMemoryBuffer tryAllocate(long bytes, boolean stageDeviceToHost) {
    if (bytes <= 0) {
      throw new IllegalArgumentException("bytes must be positive");
    }
    poolAllocCalls.incrementAndGet();
    poolAllocBytes.addAndGet(bytes);
    updateMax(maxPoolAllocBytes, bytes);
    HostMemoryBuffer result = null;
    PageableMemoryPool pool = getSingleton();
    if (pool != null) {
      result = pool.tryAllocateInternal(bytes, stageDeviceToHost);
    }
    if (result == null) {
      poolAllocMiss.incrementAndGet();
      poolAllocMissBytes.addAndGet(bytes);
      updateMax(maxPoolAllocMissBytes, bytes);
    } else {
      poolAllocSuccess.incrementAndGet();
      long live = poolLiveBytes.addAndGet(bytes);
      updateMax(poolMaxLiveBytes, live);
    }
    return result;
  }

  /**
   * Get the number of bytes that the pageable memory pool was allocated with.
   */
  public static long getTotalPoolSizeBytes() {
    PageableMemoryPool pool = getSingleton();
    if (pool != null) {
      return pool.poolSize;
    }
    return 0;
  }

  public static boolean isPageablePoolBuffer(HostMemoryBuffer buffer) {
    return buffer.cleaner instanceof PageableHostBufferCleaner;
  }

  private static boolean isStagedDeviceToHostBuffer(HostMemoryBuffer buffer) {
    return buffer.cleaner instanceof RawStagedHostBufferCleaner ||
        (buffer.cleaner instanceof PageableHostBufferCleaner &&
        ((PageableHostBufferCleaner) buffer.cleaner).stageDeviceToHost);
  }

  /** Copy device data into a pageable host destination through pinned staging buffers. */
  public static void copyDeviceToPageableHost(HostMemoryBuffer dst, long dstOffset,
      BaseDeviceMemoryBuffer src, long srcOffset, long length, Cuda.Stream stream) {
    dst.addressOutOfBoundsCheck(dst.address + dstOffset, length, "copy range dest");
    src.addressOutOfBoundsCheck(src.address + srcOffset, length, "copy range src");
    if (length > 0 && !copyFromDeviceBuffer(dst, dstOffset, src, srcOffset, length,
        stream, true)) {
      throw new IllegalStateException("Staged D2H requires initialized pinned staging buffers");
    }
  }

  /** Restore pageable host data through pinned staging buffers. */
  public static void copyPageableHostToDevice(BaseDeviceMemoryBuffer dst, long dstOffset,
      HostMemoryBuffer src, long srcOffset, long length, Cuda.Stream stream) {
    dst.addressOutOfBoundsCheck(dst.address + dstOffset, length, "copy range dest");
    src.addressOutOfBoundsCheck(src.address + srcOffset, length, "copy range src");
    if (length > 0 && !copyFromHostBuffer(dst, dstOffset, src, srcOffset, length,
        stream, true)) {
      throw new IllegalStateException("Staged H2D requires a pageable pool source " +
          "and initialized pinned staging buffers");
    }
  }

  private static void closeStagedCopyBuffers() {
    HostMemoryBuffer[] buffers = stagedCopyBuffers;
    stagedCopyBuffers = null;
    if (buffers != null) {
      for (HostMemoryBuffer buffer : buffers) {
        if (buffer != null) {
          buffer.close();
        }
      }
    }
    long[] registered = registeredStagedCopyBuffers;
    registeredStagedCopyBuffers = null;
    if (registered != null) {
      for (long address : registered) {
        if (address != 0) {
          freeRegisteredStagingBuffer(address);
        }
      }
    }
  }

  static boolean copyFromDeviceBuffer(HostMemoryBuffer dst, long dstOffset,
      BaseDeviceMemoryBuffer src, long srcOffset, long length, Cuda.Stream stream) {
    return copyFromDeviceBuffer(dst, dstOffset, src, srcOffset, length, stream,
        isStagedDeviceToHostBuffer(dst));
  }

  private static boolean copyFromDeviceBuffer(HostMemoryBuffer dst, long dstOffset,
      BaseDeviceMemoryBuffer src, long srcOffset, long length, Cuda.Stream stream,
      boolean explicitStage) {
    stagedCopyLifecycleLock.readLock().lock();
    try {
      return copyFromDeviceBufferWithLease(dst, dstOffset, src, srcOffset, length,
          stream, explicitStage);
    } finally {
      stagedCopyLifecycleLock.readLock().unlock();
    }
  }

  private static boolean copyFromDeviceBufferWithLease(HostMemoryBuffer dst, long dstOffset,
      BaseDeviceMemoryBuffer src, long srcOffset, long length, Cuda.Stream stream,
      boolean explicitStage) {
    if (length <= 0) {
      return false;
    }
    if (!explicitStage && (Boolean.getBoolean("ai.rapids.cudf.pageable.staged.copy.disabled") ||
        Boolean.getBoolean("ai.rapids.cudf.pageable.staged.copy.d2h.disabled"))) {
      stagedCopySkippedDisabledCount.incrementAndGet();
      return false;
    }
    if (!explicitStage && !isPageablePoolBuffer(dst)) {
      stagedCopySkippedNonPoolCount.incrementAndGet();
      return false;
    }
    if (!explicitStage && length < stagedCopyMinSize) {
      stagedCopySkippedSmallCount.incrementAndGet();
      return false;
    }
    ExecutorService executor = stagedCopyExecutor;
    if (executor == null) {
      stagedCopySkippedNoExecutorCount.incrementAndGet();
      return false;
    }
    long totalStart = System.nanoTime();
    long inFlight = stagedCopyInFlight.incrementAndGet();
    updateMax(stagedCopyMaxInFlight, inFlight);
    long callNumber = stagedCopyCallCount.incrementAndGet();
    stagedD2HCalls.incrementAndGet();
    stagedD2HBytes.addAndGet(length);
    stagedCopyBytes.addAndGet(length);
    updateMax(stagedCopyMaxBytes, length);
    Cuda.Event ready = null;
    NvtxRange stagedRange = null;
    try {
      ready = new Cuda.Event(false, true);
      stagedRange = new NvtxRange("pageableStagedD2H", NvtxColor.ORANGE);
      long recordStart = System.nanoTime();
      ready.record(stream);
      stagedCopyRecordEventNs.addAndGet(System.nanoTime() - recordStart);
      if (asyncStagedD2H) {
        copyFromDeviceAsync(dst.address + dstOffset, src.address + srcOffset, length, ready);
        return true;
      }
      int chunks = (int) Math.min(stagedCopyThreads,
          (length + stagedCopyChunkSize - 1) / stagedCopyChunkSize);
      stagedCopyChunks.addAndGet(chunks);
      List<Future<?>> futures = new ArrayList<>(chunks);
      long remaining = length;
      long copyOffset = 0;
      long submitStart = System.nanoTime();
      final String destinationSource = dst.cleaner instanceof PageableHostBufferCleaner ?
          "pool" : dst.cleaner instanceof RawStagedHostBufferCleaner ? "raw" : "other";
      try {
        while (remaining > 0) {
          long chunk = Math.min(stagedCopyChunkSize, remaining);
          long dstAddress = dst.address + dstOffset + copyOffset;
          long srcAddress = src.address + srcOffset + copyOffset;
          long waitEvent = ready.getEvent();
          long queuedAt = System.nanoTime();
          futures.add(executor.submit(() -> {
            long queueNs = System.nanoTime() - queuedAt;
            stagedCopyQueueNs.addAndGet(queueNs);
            updateMax(stagedCopyMaxQueueNs, queueNs);
            long active = stagedCopyWorkerActive.incrementAndGet();
            updateMax(stagedCopyMaxWorkerActive, active);
            try (NvtxRange chunkRange =
                     new NvtxRange(TRACE_STAGED_COPY_CHUNKS ?
                         "pageableStagedD2HChunk source=" + destinationSource +
                             " bytes=" + chunk + " queueWaitNs=" + queueNs :
                         "pageableStagedD2HChunk", NvtxColor.YELLOW)) {
              stagedCopyFromDevice(dstAddress, srcAddress, chunk, waitEvent);
            } finally {
              stagedCopyWorkerActive.decrementAndGet();
            }
          }));
          copyOffset += chunk;
          remaining -= chunk;
        }
      } finally {
        stagedCopySubmitNs.addAndGet(System.nanoTime() - submitStart);
        long waitStart = System.nanoTime();
        waitForStagedCopies(futures);
        stagedCopyWaitNs.addAndGet(System.nanoTime() - waitStart);
      }
      return true;
    } finally {
      stagedCopyTotalNs.addAndGet(System.nanoTime() - totalStart);
      stagedCopyInFlight.decrementAndGet();
      if (ready != null) {
        ready.close();
      }
      if (stagedRange != null) {
        stagedRange.close();
      }
      if (callNumber == 1 || callNumber % 100 == 0) {
        logStagedCopyStats();
      }
    }
  }

  static boolean copyFromHostBuffer(BaseDeviceMemoryBuffer dst, long dstOffset,
      HostMemoryBuffer src, long srcOffset, long length, Cuda.Stream stream) {
    return copyFromHostBuffer(dst, dstOffset, src, srcOffset, length, stream, false);
  }

  private static boolean copyFromHostBuffer(BaseDeviceMemoryBuffer dst, long dstOffset,
      HostMemoryBuffer src, long srcOffset, long length, Cuda.Stream stream,
      boolean explicitStage) {
    stagedCopyLifecycleLock.readLock().lock();
    try {
      return copyFromHostBufferWithLease(dst, dstOffset, src, srcOffset, length,
          stream, explicitStage);
    } finally {
      stagedCopyLifecycleLock.readLock().unlock();
    }
  }

  private static boolean copyFromHostBufferWithLease(BaseDeviceMemoryBuffer dst, long dstOffset,
      HostMemoryBuffer src, long srcOffset, long length, Cuda.Stream stream,
      boolean explicitStage) {
    if (length <= 0) {
      return false;
    }
    if (!explicitStage && Boolean.getBoolean("ai.rapids.cudf.pageable.staged.copy.h2d.disabled")) {
      stagedCopySkippedDisabledCount.incrementAndGet();
      return false;
    }
    if (!explicitStage && Boolean.getBoolean("ai.rapids.cudf.pageable.staged.copy.disabled")) {
      stagedCopySkippedDisabledCount.incrementAndGet();
      return false;
    }
    if (!isPageablePoolBuffer(src)) {
      stagedCopySkippedNonPoolCount.incrementAndGet();
      return false;
    }
    if (!explicitStage && length < stagedCopyMinSize) {
      stagedCopySkippedSmallCount.incrementAndGet();
      return false;
    }
    ExecutorService executor = stagedCopyExecutor;
    if (executor == null) {
      stagedCopySkippedNoExecutorCount.incrementAndGet();
      return false;
    }
    long totalStart = System.nanoTime();
    long inFlight = stagedCopyInFlight.incrementAndGet();
    updateMax(stagedCopyMaxInFlight, inFlight);
    long callNumber = stagedCopyCallCount.incrementAndGet();
    stagedH2DCalls.incrementAndGet();
    stagedH2DBytes.addAndGet(length);
    stagedCopyBytes.addAndGet(length);
    updateMax(stagedCopyMaxBytes, length);
    Cuda.Event ready = null;
    NvtxRange stagedRange = null;
    try {
      ready = new Cuda.Event(false, true);
      stagedRange = new NvtxRange("pageableStagedH2D", NvtxColor.CYAN);
      long recordStart = System.nanoTime();
      ready.record(stream);
      stagedCopyRecordEventNs.addAndGet(System.nanoTime() - recordStart);
      int chunks = (int) Math.min(stagedCopyThreads,
          (length + stagedCopyChunkSize - 1) / stagedCopyChunkSize);
      stagedCopyChunks.addAndGet(chunks);
      List<Future<?>> futures = new ArrayList<>(chunks);
      long remaining = length;
      long copyOffset = 0;
      long submitStart = System.nanoTime();
      try {
        while (remaining > 0) {
          long chunk = Math.min(stagedCopyChunkSize, remaining);
          long dstAddress = dst.address + dstOffset + copyOffset;
          long srcAddress = src.address + srcOffset + copyOffset;
          long waitEvent = ready.getEvent();
          long queuedAt = System.nanoTime();
          futures.add(executor.submit(() -> {
            long queueNs = System.nanoTime() - queuedAt;
            stagedCopyQueueNs.addAndGet(queueNs);
            updateMax(stagedCopyMaxQueueNs, queueNs);
            try (NvtxRange chunkRange =
                     new NvtxRange(TRACE_STAGED_COPY_CHUNKS ?
                         "pageableStagedH2DChunk source=pool bytes=" + chunk +
                             " queueWaitNs=" + queueNs :
                         "pageableStagedH2DChunk", NvtxColor.BLUE)) {
              stagedCopyToDevice(dstAddress, srcAddress, chunk, waitEvent);
            }
          }));
          copyOffset += chunk;
          remaining -= chunk;
        }
      } finally {
        stagedCopySubmitNs.addAndGet(System.nanoTime() - submitStart);
        long waitStart = System.nanoTime();
        waitForStagedCopies(futures);
        stagedCopyWaitNs.addAndGet(System.nanoTime() - waitStart);
      }
      return true;
    } finally {
      stagedCopyTotalNs.addAndGet(System.nanoTime() - totalStart);
      stagedCopyInFlight.decrementAndGet();
      if (ready != null) {
        ready.close();
      }
      if (stagedRange != null) {
        stagedRange.close();
      }
      if (callNumber == 1 || callNumber % 100 == 0) {
        logStagedCopyStats();
      }
    }
  }

  private static void updateMax(AtomicLong maxValue, long value) {
    long oldValue;
    do {
      oldValue = maxValue.get();
      if (value <= oldValue) {
        return;
      }
    } while (!maxValue.compareAndSet(oldValue, value));
  }

  public static void logStagedCopyStats() {
    if (!Boolean.getBoolean("ai.rapids.cudf.pageable.staged.copy.stats.log.enabled")) {
      return;
    }
    long calls = stagedCopyCallCount.getAndSet(0);
    long d2hCalls = stagedD2HCalls.getAndSet(0);
    long d2hBytes = stagedD2HBytes.getAndSet(0);
    long h2dCalls = stagedH2DCalls.getAndSet(0);
    long h2dBytes = stagedH2DBytes.getAndSet(0);
    long skippedSmall = stagedCopySkippedSmallCount.getAndSet(0);
    long skippedDisabled = stagedCopySkippedDisabledCount.getAndSet(0);
    long skippedNonPool = stagedCopySkippedNonPoolCount.getAndSet(0);
    long skippedNoExecutor = stagedCopySkippedNoExecutorCount.getAndSet(0);
    long bytes = stagedCopyBytes.getAndSet(0);
    long chunks = stagedCopyChunks.getAndSet(0);
    long totalNs = stagedCopyTotalNs.getAndSet(0);
    long recordEventNs = stagedCopyRecordEventNs.getAndSet(0);
    long submitNs = stagedCopySubmitNs.getAndSet(0);
    long waitNs = stagedCopyWaitNs.getAndSet(0);
    long queueNs = stagedCopyQueueNs.getAndSet(0);
    long maxQueueNs = stagedCopyMaxQueueNs.getAndSet(0);
    long maxWorkerActive = stagedCopyMaxWorkerActive.getAndSet(0);
    long asyncSlotWaitNs = asyncD2HSlotWaitNs.getAndSet(0);
    long asyncMaxSlotWaitNs = asyncD2HMaxSlotWaitNs.getAndSet(0);
    long asyncCompletionPolls = asyncD2HCompletionPolls.getAndSet(0);
    long maxBytes = stagedCopyMaxBytes.getAndSet(0);
    long maxInFlight = stagedCopyMaxInFlight.getAndSet(0);
    long inFlight = stagedCopyInFlight.get();
    String nativeStats = getAndResetStagedCopyNativeStats();
    double totalMs = totalNs / 1_000_000.0;
    double gb = bytes / 1_000_000_000.0;
    double gbps = totalNs == 0 ? 0.0 : bytes / (double) totalNs;
    log.warn("Pageable staged copy stats: calls={} d2hCalls={} d2hBytes={} " +
            "h2dCalls={} h2dBytes={} skippedSmall={} skippedDisabled={} " +
            "skippedNonPool={} skippedNoExecutor={} bytes={} ({}) chunks={} " +
            "avgBytes={} maxBytes={} totalMs={} aggregateGBps={} recordEventMs={} " +
            "submitMs={} waitMs={} queueMs={} maxQueueMs={} maxWorkerActive={} " +
            "maxInFlight={} inFlight={} asyncD2H={} asyncSlotWaitMs={} " +
            "asyncMaxSlotWaitMs={} asyncCompletionPolls={} asyncD2HSlots={} " +
            "minBytes={} chunkBytes={} " +
            "threads={} registeredBuffers={} poolAllocCalls={} poolAllocSuccess={} poolAllocMiss={} " +
            "poolAllocBytes={} poolAllocMissBytes={} maxPoolAllocBytes={} " +
            "maxPoolAllocMissBytes={} native=[{}]",
        calls, d2hCalls, d2hBytes, h2dCalls, h2dBytes,
        skippedSmall, skippedDisabled, skippedNonPool, skippedNoExecutor, bytes, gb,
        chunks, calls == 0 ? 0 : bytes / calls, maxBytes,
        totalMs, gbps, recordEventNs / 1_000_000.0, submitNs / 1_000_000.0,
        waitNs / 1_000_000.0, queueNs / 1_000_000.0, maxQueueNs / 1_000_000.0,
        maxWorkerActive, maxInFlight, inFlight, asyncStagedD2H,
        asyncSlotWaitNs / 1_000_000.0, asyncMaxSlotWaitNs / 1_000_000.0,
        asyncCompletionPolls, asyncD2HSlotCount, stagedCopyMinSize,
        stagedCopyChunkSize, stagedCopyThreads,
        Boolean.getBoolean("ai.rapids.cudf.pageable.staged.copy.registered.buffers"),
        poolAllocCalls.get(), poolAllocSuccess.get(),
        poolAllocMiss.get(), poolAllocBytes.get(), poolAllocMissBytes.get(),
        maxPoolAllocBytes.get(), maxPoolAllocMissBytes.get(), nativeStats);
  }

  public static void logPoolState() {
    PageableMemoryPool pool = getSingleton();
    String nativePoolState = pool == null ? "uninitialized" :
        "poolBytes=" + pool.poolSize;
    log.warn("Pageable pool state: liveBytes={} maxLiveBytes={} allocCalls={} " +
            "allocSuccess={} allocMiss={} allocBytes={} allocMissBytes={} " +
            "maxAllocBytes={} maxAllocMissBytes={} native=[{}]",
        poolLiveBytes.get(), poolMaxLiveBytes.get(), poolAllocCalls.get(),
        poolAllocSuccess.get(), poolAllocMiss.get(), poolAllocBytes.get(),
        poolAllocMissBytes.get(), maxPoolAllocBytes.get(), maxPoolAllocMissBytes.get(),
        nativePoolState);
  }

  static void waitForStagedCopies(List<Future<?>> futures) {
    Throwable first = null;
    boolean interrupted = false;
    for (Future<?> future : futures) {
      while (true) {
        try {
          future.get();
          break;
        } catch (InterruptedException e) {
          interrupted = true;
          if (first == null) {
            first = e;
          }
        } catch (ExecutionException e) {
          if (first == null) {
            first = e.getCause();
          }
          break;
        }
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
    if (first != null) {
      throw new RuntimeException("Staged pageable DtoH copy failed", first);
    }
  }

  private PageableMemoryPool(long poolSize, int pretouchThreads,
      boolean numaBind, boolean thpDeny) {
    this.poolHandle = newPageablePoolMemoryResource(
        poolSize, pretouchThreads, numaBind, thpDeny);
    this.poolSize = poolSize;
  }

  @Override
  public void close() {
    lifecycleLock.writeLock().lock();
    try {
      if (this.poolHandle != -1) {
        releasePageablePoolMemoryResource(this.poolHandle);
        this.poolHandle = -1;
      }
    } finally {
      lifecycleLock.writeLock().unlock();
    }
  }

  /**
   * Attempts to allocate from the pageable pool. Returns null rather than throwing if
   * the pool is exhausted, so callers can fall back gracefully.
   */
  private HostMemoryBuffer tryAllocateInternal(long bytes,
      boolean stageDeviceToHost) {
    long allocated;
    lifecycleLock.readLock().lock();
    try {
      if (this.poolHandle == -1) {
        throw new IllegalStateException("Pageable memory pool has been closed");
      }
      allocated = allocFromPageablePool(this.poolHandle, bytes);
    } finally {
      lifecycleLock.readLock().unlock();
    }
    if (allocated == -1) {
      return null;
    } else {
      return new HostMemoryBuffer(allocated, bytes,
              new PageableHostBufferCleaner(allocated, bytes, stageDeviceToHost));
    }
  }

  private void free(long address, long size) {
    lifecycleLock.readLock().lock();
    try {
      if (this.poolHandle == -1) {
        throw new IllegalStateException("Pageable memory pool has been closed");
      }
      freeFromPageablePool(this.poolHandle, address, size);
    } finally {
      lifecycleLock.readLock().unlock();
    }
  }
}
