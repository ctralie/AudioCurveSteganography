onmessage = function(event) {
    const samples = event.data.samples;
    const win = event.data.win;
    const LT = event.data.LT;
    const shift = event.data.shift;
    const L = event.data.L;
    const Q = event.data.Q;
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
    //let M = Math.floor((samples.length-shift)/win);
    let M = Math.floor((samples.length+LT-shift)/(win+LT));
    let SM = []; // Magnitudes
    let SP = []; // Phases
    for (let i = 0; i < freqs.length; i++) {
        let SMi = new Float32Array(M);
        let SPi = new Float32Array(M);
        for (let j = 0; j < M; j++) {
            let cosSum = 0;
            let sinSum = 0;
            for (let k = 0; k < win; k++) {
                cosSum += samples[shift+j*(win+LT)+k]*cos[i][k];
                sinSum += samples[shift+j*(win+LT)+k]*sin[i][k];
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

    // Step 4: Normalize and scale by phase
    postMessage({"type":"newTask", "taskString":"Normalizing"});
    // Step 4a: Rescale each coordinate to [-1, 1]
    for (let i = 0; i < freqs.length; i++) {
        let min = coords[i][0];
        for (let j = 0; j < coords[i].length; j++) {
            if (coords[i][j] < min) {
                min = coords[i][j];
            }
        }
        let max = coords[i][0];
        for (let j = 0; j < coords[i].length; j++) {
            coords[i][j] -= min;
            if (coords[i][j] > max) {
                max = coords[i][j];
            }
        }
        for (let j = 0; j < coords[i].length; j++) {
            coords[i][j] = coords[i][j]*2/max - 1;
        }
    }
    // Step 4b: Compute means of each phase
    const inc = 2*Math.PI/Q;
    let scales = new Float32Array(freqs.length);
    for (let i = 0; i < freqs.length; i++) {
        scales[i] = 0;
        for (let j = 0; j < SP[i].length; j++) {
            let p = SP[i][j];
            let plow = 2*(p - inc*Math.floor(p/inc))/inc;
            let phigh = 2*(inc*Math.ceil(p/inc) - p)/inc;
            p = Math.min(plow, phigh);
            scales[i] += p;
        }
        scales[i] /= SP[i].length;
        scales[i] = (scales[i]-0.25)*2;
    }
    // Step 4c: Perform final scaling
    let maxScaleIdx = 0;
    for (let i = 1; i < scales.length; i++) {
        if (scales[i] > scales[maxScaleIdx]) {
            maxScaleIdx = i;
        }
    }
    for (let i = 0; i < scales.length; i++) {
        for (let j = 0; j < coords[i].length; j++) {
            coords[i][j] *= scales[i]/scales[maxScaleIdx];
        }
    }
    

    // Return results
    postMessage({"type":"end", "coords":coords});
    
}
