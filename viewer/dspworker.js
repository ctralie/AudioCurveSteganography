onmessage = function(event) {
    const samples = event.data.samples;
    const win = event.data.win;
    const L = event.data.L;
    const freqs = event.data.freqs;
    // Step 1: Setup cosine and sine windows
    postMessage({"type":"newTask", "taskString":"Setting up windows"});
    let cos = [];
    let sin = [];
    for (let i = 0; i < freqs.length; i++) {
        let cosi = new Float32Array(win);
        let sini = new Float32Array(win);
        for (let j = 0; j < win; j++) {
            cosi[j] = Math.cos(2*Math.PI*freqs[i]*j/win);
            sini[j] = Math.sin(2*Math.PI*freqs[i]*j/win);
        }
        cos.push(cosi);
        sin.push(sini);
    }

    // Step 2: Compute spectrogram at chosen bins
    postMessage({"type":"newTask", "taskString":"Computing spectrogram"});
    let M = Math.floor(samples.length/win);
    let SM = []; // Magnitudes
    let SP = []; // Phases
    for (let i = 0; i < freqs.length; i++) {
        let SMi = new Float32Array(M);
        let SPi = new Float32Array(M);
        for (let j = 0; j < M; j++) {
            let cosSum = 0;
            let sinSum = 0;
            for (let k = 0; k < win; k++) {
                cosSum += samples[j*win+k]*cos[i][k];
                sinSum += samples[j*win+k]*sin[i][k];
            }
            SMi[j] = Math.sqrt(cosSum*cosSum + sinSum*sinSum);
            SPi[j] = Math.atan2(sinSum, cosSum);
        }
        SM.push(SMi);
        SP.push(SPi);
    }

    // Step 3: Compute sliding window using cumulative sums
    postMessage({"type":"newTask", "taskString":"Computing sliding window of magnitudes for " + freqs.length + " frequencies"});
    let coords = [];
    M = SM[0].length;
    for (let i = 0; i < freqs.length; i++) {
        let N = M-L+1;
        let coordsi = new Float32Array(N);
        let cumu = new Float32Array(M+1);
        cumu[0] = 0;
        for (let j = 0; j < M; j++) {
            cumu[j+1] = cumu[j] + SM[i][j];
        }
        for (let j = 0; j < N; j++) {
            coordsi[j] = cumu[j+L]-cumu[j];
        }
        coords.push(coordsi);
    }

    // Return results
    postMessage({"type":"end", "coords":coords});
    
}
