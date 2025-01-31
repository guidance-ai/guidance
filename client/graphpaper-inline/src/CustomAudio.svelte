<script>
  export let audioData; // Receiving base64 data from parent

  let audio;
  let isPlaying = false;
  let progress = 0;
  let duration = 0;
  let currentTime = 0;
  let volume = 1;

  function togglePlay() {
    if (audio.paused) {
      audio.play();
      isPlaying = true;
    } else {
      audio.pause();
      isPlaying = false;
    }
  }

  function updateProgress() {
    if (audio) {
      progress = (audio.currentTime / audio.duration) * 100;
      currentTime = audio.currentTime;
      duration = audio.duration || 0;
    }
  }

  function seek(event) {
    const seekBar = event.currentTarget;
    const seekPosition = (event.offsetX / seekBar.offsetWidth) * audio.duration;
    audio.currentTime = seekPosition;
  }

  function changeVolume(event) {
    volume = event.target.value;
    audio.volume = volume;
  }

  function formatTime(seconds) {
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    return `${min}:${sec < 10 ? "0" : ""}${sec}`;
  }
</script>

<div class="audio-container">
  <div class="play-button" on:click={togglePlay}>
    {#if isPlaying}
      <svg viewBox="0 0 24 24"
        ><rect x="6" y="5" width="4" height="14"></rect><rect
          x="14"
          y="5"
          width="4"
          height="14"
        ></rect></svg
      >
    {:else}
      <svg viewBox="0 0 24 24"><polygon points="5,3 19,12 5,21"></polygon></svg>
    {/if}
  </div>

  <div class="seek-bar-container" on:click={seek}>
    <div class="seek-bar" style="width: {progress}%"></div>
  </div>

  <div class="time-display">
    {formatTime(currentTime)} / {formatTime(duration)}
  </div>

  <div class="volume-control">
    <svg viewBox="0 0 24 24" width="20" height="20">
      <path
        d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.07c1.48-.74 2.5-2.26 2.5-4.04z"
      ></path>
    </svg>
    <input
      type="range"
      min="0"
      max="1"
      step="0.01"
      bind:value={volume}
      on:input={changeVolume}
    />
  </div>

  <audio
    bind:this={audio}
    on:timeupdate={updateProgress}
    src={"data:audio/wav;base64," + audioData}
  ></audio>
</div>

<style>
  .audio-container {
    border-radius: 10px;
    border: 1px solid gray;
    background: white;
    padding: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
    width: 400px;
  }

  .play-button {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    background-color: #6c7a89;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
  }

  .play-button svg {
    fill: white;
    width: 20px;
    height: 20px;
  }

  .seek-bar-container {
    flex-grow: 1;
    height: 6px;
    background: #ddd;
    border-radius: 3px;
    cursor: pointer;
    position: relative;
  }

  .seek-bar {
    height: 100%;
    background: #2979ff;
    border-radius: 3px;
    width: 0%;
    position: absolute;
  }

  .time-display {
    font-size: 14px;
    color: #555;
    min-width: 50px;
  }

  .volume-control {
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .volume-control input {
    width: 60px;
  }
</style>
