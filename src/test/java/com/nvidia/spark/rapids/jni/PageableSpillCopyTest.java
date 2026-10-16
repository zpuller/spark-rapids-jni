/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.PageableMemoryPool;
import ai.rapids.cudf.PinnedMemoryPool;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

public class PageableSpillCopyTest {
  @Test
  public void roundTripWithOffsetsAndMultipleChunks() {
    roundTrip(false, 2, false);
  }

  @Test
  public void asyncRoundTripWithOffsetsAndMultipleChunks() {
    roundTrip(true, 2, true);
  }

  @Test
  public void asyncRoundTripWithMoreSlotsThanWorkers() {
    roundTrip(true, 4, true);
  }

  private long stagedCalls() {
    try {
      java.lang.reflect.Field field = PageableMemoryPool.class.getDeclaredField("stagedCopyCallCount");
      field.setAccessible(true);
      return ((java.util.concurrent.atomic.AtomicLong) field.get(null)).get();
    } catch (ReflectiveOperationException e) {
      throw new AssertionError(e);
    }
  }

  private void roundTrip(boolean asyncD2H, int slots, boolean registeredBuffers) {
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h",
        Boolean.toString(asyncD2H));
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h.slots",
        Integer.toString(slots));
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.threads", "2");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes", "1048576");
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers",
        Boolean.toString(registeredBuffers));
    System.setProperty("ai.rapids.cudf.pageable.staged.copy.disabled", "true");
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, null, 64L * 1024 * 1024);
    try {
      if (!registeredBuffers) {
        PinnedMemoryPool.initialize(4L * 1024 * 1024);
      }
      PageableMemoryPool.initialize(8L * 1024 * 1024, 2);
      int length = 1024 * 1024 + 17;
      byte[] expected = new byte[length];
      Arrays.fill(expected, (byte) 0x5a);
      try (HostMemoryBuffer input = HostMemoryBuffer.allocateRaw(length);
           DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(length);
           DeviceMemoryBuffer restored = DeviceMemoryBuffer.allocate(length);
           HostMemoryBuffer output = HostMemoryBuffer.allocateRaw(length)) {
        input.setBytes(0, expected, 0, length);
        device.copyFromHostBuffer(input);
        try (HostMemoryBuffer staged =
                 PageableMemoryPool.tryAllocateForDeviceToHost(length)) {
          assertNotNull(staged);
          staged.copyFromDeviceBufferAsync(device, Cuda.DEFAULT_STREAM);
          byte[] automatic = new byte[length];
          staged.getBytes(automatic, 0, 0, length);
          assertArrayEquals(expected, automatic);
        }
        long callsBefore = stagedCalls();
        PageableMemoryPool.copyDeviceToPageableHost(output, 0, device, 0, length,
            Cuda.DEFAULT_STREAM);
        assertEquals(callsBefore + 1, stagedCalls());
        byte[] rawExplicit = new byte[length];
        output.getBytes(rawExplicit, 0, 0, length);
        assertArrayEquals(expected, rawExplicit);
        try (HostMemoryBuffer raw = PageableMemoryPool.allocateRawForDeviceToHost(length)) {
          assertFalse(PageableMemoryPool.isPageablePoolBuffer(raw));
          callsBefore = stagedCalls();
          raw.copyFromDeviceBufferAsync(device, Cuda.DEFAULT_STREAM);
          assertEquals(callsBefore + 1, stagedCalls());
          byte[] rawAutomatic = new byte[length];
          raw.getBytes(rawAutomatic, 0, 0, length);
          assertArrayEquals(expected, rawAutomatic);
        }
        try (HostMemoryBuffer spill = PageableMemoryPool.tryAllocate(length + 32)) {
          assertNotNull(spill);
          PageableMemoryPool.copyDeviceToPageableHost(spill, 16, device, 0, length,
              Cuda.DEFAULT_STREAM);
          PageableMemoryPool.copyPageableHostToDevice(restored, 0, spill, 16, length,
              Cuda.DEFAULT_STREAM);
        }
        output.copyFromDeviceBuffer(restored);
        byte[] actual = new byte[length];
        output.getBytes(actual, 0, 0, length);
        assertArrayEquals(expected, actual);
      }
    } finally {
      PageableMemoryPool.shutdown();
      if (!registeredBuffers) {
        PinnedMemoryPool.shutdown();
      }
      Rmm.shutdown();
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.threads");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.chunk.bytes");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.registered.buffers");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.disabled");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h");
      System.clearProperty("ai.rapids.cudf.pageable.staged.copy.async.d2h.slots");
    }
  }
}
