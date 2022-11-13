let tuneInput = document.getElementById('tuneInput');
let audioPlayer = document.getElementById("audioPlayer");
let audio = new SampledAudio(audioPlayer);
const canvasArea = document.getElementById("canvasArea");

tuneInput.addEventListener('change', function(e) {
    let reader = new FileReader();
    reader.onload = function(e) {
        audio.setSamplesAudioBuffer(e.target.result, canvasArea);
    }
    reader.readAsArrayBuffer(tuneInput.files[0]);
});

audio.loadFile("baby.mp3", canvasArea);
