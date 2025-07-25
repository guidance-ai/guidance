<script lang="ts">
  import { onMount } from "svelte";
  import type { MediaNode } from "./interfaces";

  // Add proper TypeScript types
  export let audioData: MediaNode; // Base64 data (without the data URL header)

  let audio: HTMLAudioElement;
  let isPlaying: boolean = false;
  let progress: number = 0;
  let duration: number = 0;
  let currentTime: number = 0;
  let volume: number = 1;
  let isMuted: boolean = false;
  let showVolumeSlider: boolean = false;
  let waveformCanvas: HTMLCanvasElement;

  // Store waveform data globally so we don't have to recompute it
  let waveformData: any[] = [];
  let maxAmp = 0;

  // Decode the audio, downsample it, and draw the waveform onto the canvas.
  // Track mouse position for optional hover preview
  let hoverPosition = -1;

  function togglePlay() {
    if (audio.paused) {
      audio.play();
      isPlaying = true;
    } else {
      audio.pause();
      isPlaying = false;
    }
  }

  function seek(event: MouseEvent) {
    const container = event.currentTarget as HTMLElement;
    if (container == null) {
      console.error("Null seek event target");
      return;
    }
    const seekPosition =
      (event.offsetX / container.offsetWidth) * audio.duration;
    audio.currentTime = seekPosition;
  }

  function changeVolume(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target == null) {
      console.error("Null change volume event target");
      return;
    }
    volume = parseFloat(target.value);

    // If we're adjusting volume, we're unmuting
    if (isMuted && volume > 0) {
      isMuted = false;
    }

    // Apply volume or mute
    audio.volume = isMuted ? 0 : volume;
  }

  $: if (audio) {
    // Reactively update audio volume when isMuted changes
    audio.volume = isMuted ? 0 : volume;
  }

  function formatTime(seconds: number) {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    return `${min}:${sec < 10 ? "0" : ""}${sec}`;
  }

  // Helper: convert base64 string to ArrayBuffer
  function base64ToArrayBuffer(base64: string) {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }

  function setHoverPosition(event: MouseEvent) {
    const container = event.currentTarget as HTMLElement;
    if (container == null) {
      console.error("Null hover event target");
      return;
    }
    hoverPosition = (event.offsetX / container.offsetWidth) * 100;
  }

  function clearHoverPosition() {
    hoverPosition = -1;
  }

  // Add these new variables
  let staticWaveformCanvas: HTMLCanvasElement;
  let hasRenderedStatic = false;
  let animationFrameId: number | null = null;

  // Modified drawWaveform function
  async function drawWaveform() {
    if (!audioData || !waveformCanvas) return;

    const canvas = waveformCanvas;
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    const width = canvas.width;
    const height = canvas.height;
    const ctx = canvas.getContext("2d");
    if (ctx == null) return;

    // Keep existing waveform data computation code
    if (waveformData.length === 0) {
      const audioContext = new AudioContext();
      const arrayBuffer = base64ToArrayBuffer(audioData.value);
      try {
        const decodedData = await audioContext.decodeAudioData(arrayBuffer);
        const rawData = decodedData.getChannelData(0); // use first channel

        // Downsample the raw data to one value per pixel
        const samples = width;
        const blockSize = Math.floor(rawData.length / samples);
        waveformData = new Array(samples);
        for (let i = 0; i < samples; i++) {
          let sum = 0;
          for (let j = 0; j < blockSize; j++) {
            sum += Math.abs(rawData[i * blockSize + j]);
          }
          waveformData[i] = sum / blockSize;
        }

        // Find maximum amplitude for normalization
        maxAmp = Math.max(...waveformData);
        if (maxAmp === 0) maxAmp = 1; // Prevent division by zero
      } catch (error) {
        console.error("Error decoding audio for waveform:", error);
        return;
      }
    }

    // Create static canvas for unplayed portions if needed
    if (!hasRenderedStatic) {
      staticWaveformCanvas = document.createElement("canvas");
      staticWaveformCanvas.width = width;
      staticWaveformCanvas.height = height;
      const staticCtx = staticWaveformCanvas.getContext("2d");

      if (staticCtx) {
        // Draw all bars in unplayed state
        const barWidth = 1.5;
        const gap = 1;
        const totalBars = Math.floor(width / (barWidth + gap));

        for (let i = 0; i < totalBars; i++) {
          const dataIndex = Math.floor((i / totalBars) * waveformData.length);
          const normalizedAmp = waveformData[dataIndex] / maxAmp;

          const barHeight = normalizedAmp * height * 0.8;
          const y = (height - barHeight) / 2;
          const x = i * (barWidth + gap);

          staticCtx.fillStyle = "#E5E5E5"; // Light gray for unplayed
          staticCtx.beginPath();
          staticCtx.roundRect(x, y, barWidth, barHeight, 1);
          staticCtx.fill();
        }
        hasRenderedStatic = true;
      }
    }

    // Clear and redraw
    ctx.clearRect(0, 0, width, height);

    // Draw static background
    if (hasRenderedStatic) {
      ctx.drawImage(staticWaveformCanvas, 0, 0);
    }

    // Calculate progress pixel and draw played portion
    const progressPixel = Math.floor((progress / 100) * width);
    const barWidth = 2;
    const gap = 1;
    const totalBars = Math.floor(width / (barWidth + gap));

    // Only draw played bars if there's actual progress
    if (progress > 0) {
      const barsToRedraw = Math.ceil(progressPixel / (barWidth + gap));

      for (let i = 0; i < barsToRedraw; i++) {
        const dataIndex = Math.floor((i / totalBars) * waveformData.length);
        const normalizedAmp = waveformData[dataIndex] / maxAmp;

        const barHeight = normalizedAmp * height * 0.8;
        const y = (height - barHeight) / 2;
        const x = i * (barWidth + gap);

        ctx.fillStyle = "#717171"; // Gray for played portion
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barHeight, 1);
        ctx.fill();
      }
    }

    // Draw progress indicator if playing
    if (progress > 0) {
      ctx.beginPath();
      ctx.moveTo(progressPixel, 0);
      ctx.lineTo(progressPixel, height);
      ctx.strokeStyle = "rgba(80, 80, 80, 0.7)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Draw hover indicator (keep as is)
    if (hoverPosition >= 0) {
      const hoverPixel = Math.floor((hoverPosition / 100) * width);
      ctx.beginPath();
      ctx.moveTo(hoverPixel, 0);
      ctx.lineTo(hoverPixel, height);
      ctx.strokeStyle = "rgba(0, 0, 0, 0.3)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  // Simplified updateProgress function
  function updateProgress() {
    if (audio) {
      progress = (audio.currentTime / audio.duration) * 100;
      currentTime = audio.currentTime;
      duration = audio.duration || 0;
    }
  }

  // New renderLoop function
  function renderLoop() {
    if (isPlaying) {
      updateProgress();
    }

    // Always draw waveform for hover effects
    drawWaveform();

    animationFrameId = requestAnimationFrame(renderLoop);
  }

  // Updated handleEnded function
  function handleEnded() {
    isPlaying = false;
    progress = 0;
    currentTime = 0;

    // Force waveform reset
    hasRenderedStatic = false;
    drawWaveform();
  }

  // Updated onMount
  onMount(() => {
    // Initial waveform drawing
    drawWaveform();

    // Start animation loop
    renderLoop();

    // Add resize observer
    const resizeObserver = new ResizeObserver(() => {
      hasRenderedStatic = false;
      drawWaveform();
    });

    if (waveformCanvas) {
      resizeObserver.observe(waveformCanvas);
    }

    return () => {
      if (waveformCanvas) {
        resizeObserver.disconnect();
      }
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  });
</script>

<div
  class="bg-white dark:bg-gray-800 px-2 py-1 w-full max-w-6xl rounded-xl shadow-sm border border-gray-100 dark:border-gray-700"
>
  <!-- Main player content -->
  <div class="flex flex-col gap-2">
    <!-- Top row with play button, volume control, and waveform -->
    <div class="flex items-center gap-1">
      <!-- Play Button -->
      <button
        class="w-6 h-6 rounded-full bg-gray-800 dark:bg-gray-200 flex items-center justify-center cursor-pointer transition-all hover:bg-gray-900 dark:hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-500 dark:focus:ring-gray-400"
        on:click={togglePlay}
        aria-label="Toggle playback"
      >
        {#if isPlaying}
          <svg class="fill-white dark:fill-gray-900 w-5 h-5" viewBox="0 0 24 24">
            <rect x="7" y="6" width="3" height="12" rx="1" />
            <rect x="14" y="6" width="3" height="12" rx="1" />
          </svg>
        {:else}
          <svg class="fill-white dark:fill-gray-900 w-5 h-5" viewBox="0 0 24 24">
            <path d="M8 5.14v14l11-7-11-7z" />
          </svg>
        {/if}
      </button>

      <!-- Volume Control (moved next to play button) -->
      <div
        class="relative"
        on:mouseenter={() => (showVolumeSlider = true)}
        on:mouseleave={() => (showVolumeSlider = false)}
        role="group"
        aria-label="Volume controls"
      >
        <!-- Volume Button -->
        <button
          class="text-gray-500 dark:text-gray-400 pl-1 py-1 hover:text-gray-700 dark:hover:text-gray-300 relative z-10"
          on:click={() => (isMuted = !isMuted)}
          aria-label={isMuted ? "Unmute" : "Mute"}
          aria-pressed={isMuted}
        >
          <svg class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
            {#if isMuted || volume === 0}
              <!-- Muted icon -->
              <path
                d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3L3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4L9.91 6.09 12 8.18V4z"
              ></path>
            {:else if volume < 0.5}
              <!-- Low volume icon -->
              <path d="M7 9v6h4l5 5V4l-5 5H7z"></path>
            {:else}
              <!-- High volume icon -->
              <path
                d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.07c1.48-.74 2.5-2.26 2.5-4.04z"
              ></path>
            {/if}
          </svg>
        </button>

        <!-- Volume Slider (appears on hover) -->
        {#if showVolumeSlider}
          <div
            class="absolute left-0 bottom-[-15px] bg-white dark:bg-gray-800 shadow-md rounded-lg p-2 transform -translate-x-1/4 transition-opacity duration-200 z-20"
            role="slider"
            aria-label="Volume"
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuenow={volume * 100}
          >
            <div class="w-24 relative h-1 rounded-full bg-gray-200 dark:bg-gray-600">
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                bind:value={volume}
                on:input={changeVolume}
                class="absolute inset-0 opacity-0 cursor-pointer z-10 w-full"
                aria-label="Volume"
              />
              <div
                class="absolute inset-y-0 left-0 rounded-full bg-gray-600 dark:bg-gray-300"
                style="width: {volume * 100}%"
              ></div>
              <div
                class="absolute h-2 w-2 bg-gray-50 dark:bg-gray-700 border border-gray-300 dark:border-gray-500 rounded-full shadow-sm"
                style="left: calc({volume * 100}% - 6px); top: -2px"
              ></div>
            </div>
          </div>
        {/if}
      </div>

      <!-- Waveform Canvas (clickable) -->
      <div
        class="flex-grow relative cursor-pointer"
        on:click={seek}
        on:mousemove={setHoverPosition}
        on:mouseleave={clearHoverPosition}
        on:keydown={(e) => {
          // Add keyboard controls for seeking
          if (e.key === "ArrowRight") {
            audio.currentTime = Math.min(audio.duration, audio.currentTime + 5);
          } else if (e.key === "ArrowLeft") {
            audio.currentTime = Math.max(0, audio.currentTime - 5);
          }
        }}
        role="slider"
        tabindex="0"
        aria-label="Audio timeline"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={progress}
      >
        <canvas bind:this={waveformCanvas} class="w-full h-12"></canvas>
      </div>

      <!-- Time Display -->
      <div class="text-gray-700 dark:text-gray-300 whitespace-nowrap text-sm">
        {formatTime(currentTime)} / {formatTime(duration)}
      </div>
    </div>
  </div>

  <!-- Hidden audio element -->
  <audio
    bind:this={audio}
    on:timeupdate={updateProgress}
    on:ended={handleEnded}
    src={`data:audio/${audioData.format};base64,` + audioData.value}
    class="hidden"
  ></audio>
</div>
