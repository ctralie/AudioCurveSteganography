const WINDOW_FAC = 0.7;

/**
 * Create audio samples in the wav format
 * @param {array} channels Array arrays of audio samples
 * @param {int} sr Sample rate
 */
function createWavURL(channels, sr) {
    const nChannels = channels.length;
    const N = channels[0].length;
    let audio = new Float32Array(N*nChannels);
    // Interleave audio channels
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < nChannels; j++) {
        audio[i*nChannels+j] = channels[j][i];
      }
    }

    // get WAV file bytes and audio params of your audio source
    const wavBytes = getWavBytes(audio.buffer, {
      isFloat: true,       // floating point or 16-bit integer
      numChannels: nChannels,
      sampleRate: sr,
    })
    const wav = new Blob([wavBytes], {type: 'audio/wav'});
    return window.URL.createObjectURL(wav);
}


class SampledAudio {
  constructor(audioPlayer) {
    this.mediaRecorder = null;
    this.audio = null;
    this.recorder = null;
    this.audioPlayer = audioPlayer;

    this.audioBlob = null;
    this.channels = [];
    this.sr = 44100;
    this.audioContext = new AudioContext({sampleRate:this.sr});
    this.setupMenu();

  }

  setupMenu() {
    let shaderObj = this;
    let menu = new dat.GUI();
    this.menu = menu;
    this.folders = [];
  }


  /**
   * Set the audio samples based on an array buffer
   * @param {ArrayBuffer} data Array buffer with audio data
   * @param {DOM Element} canvasArea Handle to the canvas div where canvases
   *                                 can be added
   * @param {object} params Default audio parameters
   * @returns 
   */
  setSamplesAudioBuffer(data, canvasArea, params) {
    let that = this;
    return new Promise(resolve => {
      that.audioContext.decodeAudioData(data, function(buff) {
        that.channels = [];
        for (let i = 0; i < buff.numberOfChannels; i++) {
          that.channels.push(buff.getChannelData(i));
        }
        that.sr = buff.sampleRate;
        that.audioPlayer.src = createWavURL(that.channels, that.sr);

        // Remove previous menus
        for (let i = 0; i < that.folders.length; i++) {
          that.menu.removeFolder(that.folders[i]);
        }
        that.folders = [];

        // Setup canvases
        canvasArea.innerHTML = ""; // Clear prior canvases

        let width = WINDOW_FAC*Math.min(window.innerWidth, window.innerHeight);
        let folder = that.menu.addFolder("Parameters");
        that.folders.push(folder);
        let canvas = document.createElement("canvas");
        canvas.width=width;
        canvas.height=width;
        canvasArea.appendChild(canvas);
        that.stegCanvas = new StegCanvas(canvas, that.audioPlayer, that.channels[0], that.sr, folder, params);
        resolve();
      });
    });
  }

  /**
   * Load in the samples from an audio file
   * @param {string} path Path to audio file
   * @param {DOM Element} canvasArea Handle to the canvas div where canvases
   *                                 can be added
   * @param {string} colormap Default colormap to use
   * @returns A promise for when the samples have been loaded and set
   */
  loadFile(path, canvasArea, colormap) {
    let that = this;
    return new Promise((resolve, reject) => {
      $.get(path, function(data) {
        that.setSamplesAudioBuffer(data, canvasArea, colormap);
      }, "arraybuffer")
      .fail(() => {
        reject();
      });
    });
  }

  /**
   * Play the audio
   */
  playAudio() {
    this.audio.play();
  }

  /**
   * Download the audio as a WAV
   */
  downloadAudio() {
    const a = document.createElement("a");
    a.href = createWavURL(this.channels, this.sr);
    a.style.display = 'none';
    a.download = "audio.wav";
    document.body.appendChild(a);
    a.click();
  }

}