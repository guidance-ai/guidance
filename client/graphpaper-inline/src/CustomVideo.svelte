<script lang="ts">
  import "video.js/dist/video-js.css";
  import videojs from "video.js";
  import { onMount, onDestroy } from "svelte";

  export let videoData: { value: string };
  let videoElement: HTMLVideoElement;
  let player: any;

  onMount(() => {
    // Debug log
    console.log("videoElement exists?", !!videoElement);

    if (videoElement) {
      // Add a small delay to ensure DOM is ready
      setTimeout(() => {
        try {
          player = videojs(videoElement, {
            controls: true,
            fluid: true,
            playsinline: true,
            controlBar: {
                fullscreenToggle: true
            }
          });
          console.log("Player initialized successfully");
        } catch (e) {
          console.error("Failed to initialize player:", e);
        }
      }, 0);
    } else {
      console.error("Video element not found during mount");
    }
  });

  onDestroy(() => {
    if (player) {
      player.dispose();
    }
  });
</script>

<div class="video-container">
  <video bind:this={videoElement} class="video-js" playsinline allow="fullscreen" controls>
    <source src={`data:video/mp4;base64,${videoData}`} type="video/mp4" />
  </video>
</div>

<style>
  .video-container {
    width: 500px; /* Todo: make this more dynamic */
  }
</style>
