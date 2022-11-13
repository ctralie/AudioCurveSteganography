const VERT_SRC = `
attribute vec3 a_info;
attribute vec3 a_color;

// Uniforms set from Javascript that are constant
// over all fragments
uniform vec2 uCenter; // Where the origin (0, 0) is on the canvas
uniform float uScale; // Scale of fractal
uniform float uPointSize; // Size to draw points

varying float v_time;
varying vec3 v_color;

void main() {
  gl_PointSize = uPointSize;
  vec2 a_position = vec2(a_info.x, a_info.y);
  gl_Position = vec4((a_position-uCenter)*uScale, 0, 1);
  v_time = a_info.z;
  v_color = a_color;
}`;

const FRAG_SRC = `
precision highp float;

varying float v_time;
varying vec3 v_color;
uniform float uTime;

void main() {
    gl_FragColor = vec4(v_color, 1);
}`;


getMousePos = function(evt) {
    if ('touches' in evt) {
        return {
            X: evt.touches[0].clientX,
            Y: evt.touches[1].clientY
        }
    }
    return {
        X: evt.clientX,
        Y: evt.clientY
    };
}

/**
 * A function that compiles a particular shader
 * @param {object} gl WebGL handle
 * @param {string} shadersrc A string holding the GLSL source code for the shader
 * @param {string} type The type of shader ("fragment" or "vertex") 
 * 
 * @returns{shader} Shader object
 */
function getShader(gl, shadersrc, type) {
    var shader;
    if (type == "fragment") {
        shader = gl.createShader(gl.FRAGMENT_SHADER);
    } 
    else if (type == "vertex") {
        shader = gl.createShader(gl.VERTEX_SHADER);
    } 
    else {
        return null;
    }
    
    gl.shaderSource(shader, shadersrc);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.log("Unable to compile " + type + " shader...")
        console.log(shadersrc);
        console.log(gl.getShaderInfoLog(shader));
        alert("Could not compile shader");
        return null;
    }
    return shader;
}


/**
 * Compile a vertex shader and a fragment shader and link them together
 * 
 * @param {object} gl WebGL Handle
 * @param {string} prefix Prefix for naming the shader
 * @param {string} vertexSrc A string holding the GLSL source code for the vertex shader
 * @param {string} fragmentSrc A string holding the GLSL source code for the fragment shader
 */
function getShaderProgram(gl, prefix, vertexSrc, fragmentSrc) {
    let vertexShader = getShader(gl, vertexSrc, "vertex");
    let fragmentShader = getShader(gl, fragmentSrc, "fragment");
    let shader = gl.createProgram();
    gl.attachShader(shader, vertexShader);
    gl.attachShader(shader, fragmentShader);
    gl.linkProgram(shader);
    if (!gl.getProgramParameter(shader, gl.LINK_STATUS)) {
        reject(Error("Could not initialize shader" + prefix));
    }
    shader.name = prefix;
    return shader;
}

/**
 * Load in and compile a vertex/fragment shader pair asynchronously
 * 
 * @param {object} gl WebGL Handle
 * @param {string} prefix File prefix for shader.  It is expected that there
 * will be both a vertex shader named prefix.vert and a fragment
 * shader named prefix.frag
 * 
 * @returns{Promise} A promise that resolves to a shader program, where the 
 * vertex/fragment shaders are compiled/linked together
 */
function getShaderProgramAsync(gl, prefix) {
    return new Promise((resolve, reject) => {
        $.get(prefix + ".vert", function(vertexSrc) {
            $.get(prefix + ".frag", function(fragmentSrc) {
                resolve(getShaderProgram(gl, prefix, vertexSrc, fragmentSrc));
            });
        });
    });
}

class StegCanvas2D {
    /**
     * 
     * @param {DOM Element} glcanvas Handle to GL Canvas
     * @param {DOM Element} audioPlayer Handle to audio player object
     * @param {Float32Array} audioSamples Audio samples
     * @param {int} sr Sample rate
     * @param {Dat.GUI Folder} folder Handle to dat.gui folder for options
     * @param {string} colormap Default colormap to use
     */
    constructor(glcanvas, audioPlayer, audioSamples, sr, folder, colormap) {
        this.audioPlayer = audioPlayer;
        this.audioSamples = audioSamples;
        this.sr = sr;
        const that = this;
        glcanvas.addEventListener("contextmenu", function(e){ e.stopPropagation(); e.preventDefault(); return false; }); //Need this to disable the menu that pops up on right clicking
        this.colorBuffers = {};

        // Viewing options
        this.centervec = [0, 0];
        this.scale = 1;
        this.win = 1024;
        this.L = 16;
        this.Q = 4;
        this.freq1 = 1;
        this.freq2 = 2;
        
        try {
            glcanvas.gl = glcanvas.getContext("webgl");
            glcanvas.gl.viewportWidth = glcanvas.width;
            glcanvas.gl.viewportHeight = glcanvas.height;
            this.glcanvas = glcanvas;
            this.setupMenus(folder, colormap);
            this.loadShader();
            this.setupMouseHandlers();
            this.extractSignal();
            const repaint = () => {requestAnimationFrame(that.render.bind(that));};
            if (!(this.audioPlayer.onplay === null)) {
                let fn = this.audioPlayer.onplay;
                this.audioPlayer.onplay = () => {fn(); repaint()}
            }
            else {
                this.audioPlayer.onplay = repaint;
            }
            if (!(this.audioPlayer.onseeked === null)) {
                let fn = this.audioPlayer.onseeked;
                this.audioPlayer.onseeked = () => {fn(); repaint()}
            }
            else {
                this.audioPlayer.onseeked = repaint;
            }
        } catch (e) {
            alert("WebGL Error");
            console.log(e);
        }
    }

    /**
     * Setup the Dat.GUI folder for a particular canvas
     * @param {dat.gui folder} folder 
     * @param {string} colormap Default colormap to use
     */
    setupMenus(folder, colormap) {
        const that = this;
        const repaint = () => {
            if (that.audioPlayer.paused) {
                requestAnimationFrame(that.render.bind(that));
            }
        };
        this.progressBar = new ProgressBar();
        this.folder = folder;
        this.stippleSize = 0.2;
        folder.add(this, "win", 0, this.sr, 1).onChange(this.extractSignal.bind(this));
        folder.add(this, "L", 0).onChange(this.extractSignal.bind(this));
        folder.add(this, "Q", 1, 10, 1).onChange(this.extractSignal.bind(this));
        folder.add(this, "freq1", 0, Math.floor(this.win/2), 1).onChange(this.extractSignal.bind(this));
        folder.add(this, "freq2", 0, Math.floor(this.win/2), 1).onChange(this.extractSignal.bind(this));
        folder.add(this, "stippleSize", 0.1, 4).onChange(repaint);
        let colormapKeys = [];
        for (let key in colormaps) {
            if (colormaps.hasOwnProperty(key)) {
                colormapKeys.push(key);
            }
        }
        if (colormap === undefined) {
            colormap = colormapKeys[0];
        }
        this.colormap = colormap;
        folder.add(this, "colormap", colormapKeys).onChange(repaint);
    }

    /**
     * Load in the shaders from stipple.frag and stipple.vert
     */
    loadShader() {
        let gl = this.glcanvas.gl;
        let shader = getShaderProgram(gl, "stipple", VERT_SRC, FRAG_SRC);
        // Extract uniforms and store them in the shader object
        shader.uCenterUniform = gl.getUniformLocation(shader, "uCenter");
        shader.uScaleUniform = gl.getUniformLocation(shader, "uScale");
        shader.uTimeUniform = gl.getUniformLocation(shader, "uTime");
        shader.uPointSizeUniform = gl.getUniformLocation(shader, "uPointSize");
        // Extract the info buffer and store it in the shader object
        shader.infoLocation = gl.getAttribLocation(shader, "a_info");
        gl.enableVertexAttribArray(shader.infoLocation);
        // Extract the colormap buffer and store it in the shader object
        shader.colorLocation = gl.getAttribLocation(shader, "a_color");
        gl.enableVertexAttribArray(shader.colorLocation);
        this.shader = shader;
        this.infoBuffer = gl.createBuffer();
    }

    /**
     * Extract the hidden signal and fill in the coordinates
     */
    extractSignal() {
        const that = this;
        const progressBar = this.progressBar;
        new Promise((resolve, reject) => {
            const worker = new Worker("dspworker.js");
            let payload = {samples:that.audioSamples, win:that.win, L:that.L, Q:that.Q, freqs:[this.freq1, this.freq2]};
            worker.postMessage(payload);
            worker.onmessage = function(event) {
                if (event.data.type == "newTask") {
                    progressBar.loadString = event.data.taskString;
                }
                else if (event.data.type == "error") {
                    that.progressBar.setLoadingFailed(event.data.taskString);
                    reject();
                }
                else if (event.data.type == "debug") {
                    console.log("Debug: " + event.data.taskString);
                }
                else if (event.data.type == "end") {
                    let coords = event.data.coords;
                    that.x = coords[0];
                    that.y = coords[1];
                    that.colorBuffers = {};
                    that.setupBuffers();
                    resolve();
                }
                else {
                    console.log("Unknown command");
                }
            }
        }).then(() => {
            progressBar.changeToReady();
            progressBar.changeMessage("Ready!  Press play!");
        }).catch(reason => {
            progressBar.setLoadingFailed(reason);
        });
        progressBar.startLoading();
    }

    /**
     * Setup the position, color, and time buffers
     */
    setupBuffers() {
        const gl = this.glcanvas.gl;
        // Setup position and time buffers
        const N = this.x.length;
        let info = new Float32Array(N*3);
        // Finish setting up buffers
        for (let i = 0; i < N; i++) {
            info[i*3] = this.x[i];
            info[i*3+1] = this.y[i];
            info[i*3+2] = i/N;
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.infoBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, info, gl.STATIC_DRAW);
        gl.vertexAttribPointer(this.shader.infoLocation, 3, gl.FLOAT, false, 0, 0);
        this.render();
    }

    /**
     * Setup the buffer for a particular colormap
     * @param {string} key Colormap to use
     */
    setupColorBuffer(key) {
        const N = this.x.length;
        let C = new Float32Array(N*3);
        for (let i = 0; i < N; i++) {
            let idx = i*colormaps[key].length / N;
            let i1 = Math.floor(idx);
            let i2 = i1+1;
            if (i2 > colormaps[key].length-1) {
                i2 = colormaps[key].length-1;
            }
            let t = idx - i1;
            for (let k = 0; k < 3; k++) {
                // Linear interpolation of colors
                C[i*3+k] = colormaps[key][i1][k]*(1-t) + colormaps[key][i2][k]*t;
            }
        }
        const gl = this.glcanvas.gl;
        let colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, C, gl.STATIC_DRAW);
        this.colorBuffers[key] = colorBuffer;
    }

    render() {
        const gl = this.glcanvas.gl;
        gl.clearColor(1, 1, 1, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        let shader = this.shader;
        gl.useProgram(shader);
        // Step 1: Setup uniform variables that are sent to the shaders
        gl.uniform2fv(shader.uCenterUniform, this.centervec);
        gl.uniform1f(shader.uScaleUniform, this.scale);
        let t = this.audioPlayer.currentTime / this.audioPlayer.duration;
        gl.uniform1f(shader.uTime, t);

        let N = Math.floor(this.audioPlayer.currentTime*this.sr/this.win);
        N = Math.min(N, this.x.length);

        // Step 2: Bind vertex and time buffers to draw points
        gl.bindBuffer(gl.ARRAY_BUFFER, this.infoBuffer);
        gl.vertexAttribPointer(shader.infoLocation, 3, gl.FLOAT, false, 0, 0);
        if (!(this.colormap in this.colorBuffers)) {
            this.setupColorBuffer(this.colormap);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuffers[this.colormap]);
        gl.vertexAttribPointer(shader.colorLocation, 3, gl.FLOAT, false, 0, 0);

        gl.uniform1f(shader.uPointSizeUniform, this.scale*this.stippleSize);
        gl.drawArrays(gl.POINTS, 0, N);
        gl.uniform1f(shader.uPointSizeUniform, 10.0*this.scale*this.stippleSize);
        gl.drawArrays(gl.POINTS, N, 1);
        gl.drawArrays(gl.LINES, 0, N);
        gl.drawArrays(gl.LINES, 1, N-1);
        if (this.audioPlayer.duration > 0 && !this.audioPlayer.paused) {
            requestAnimationFrame(this.render.bind(this));
        }
    }
    

    /**
     * Setup functions to handle mouse events.  These may or may not
     * be used in individual shaders, but their behavior is shared across
     * many different types of shaders, so they should be available
     */
    setupMouseHandlers() {
        this.dragging = false;
        this.justClicked = false;
        this.clickType = "LEFT";
        this.lastX = 0;
        this.lastY = 0;
    
        this.glcanvas.addEventListener('mousedown', this.makeClick.bind(this));
        this.glcanvas.addEventListener('mouseup', this.releaseClick.bind(this));
        this.glcanvas.addEventListener('mousemove', this.clickerDragged.bind(this));
        this.glcanvas.addEventListener('mouseout', this.mouseOut.bind(this));
    
        //Support for mobile devices
        this.glcanvas.addEventListener('touchstart', this.makeClick.bind(this));
        this.glcanvas.addEventListener('touchend', this.releaseClick.bind(this));
        this.glcanvas.addEventListener('touchmove', this.clickerDragged.bind(this));
    }
    releaseClick(evt) {
        evt.preventDefault();
        this.dragging = false;
        return false;
    } 
    mouseOut() {
        this.dragging = false;
        return false;
    }
    makeClick(e) {
        let evt = (e == null ? event:e);
        this.clickType = "LEFT";
        evt.preventDefault();
        if (evt.which) {
            if (evt.which == 3) this.clickType = "RIGHT";
            if (evt.which == 2) this.clickType = "MIDDLE";
        }
        else if (evt.button) {
            if (evt.button == 2) this.clickType = "RIGHT";
            if (evt.button == 4) this.clickType = "MIDDLE";
        }
        this.dragging = true;
        this.justClicked = true;
        let mousePos = getMousePos(evt);
        this.lastX = mousePos.X;
        this.lastY = mousePos.Y;
        return false;
    } 
    clickerDragged(evt) {
        evt.preventDefault();
        this.clickerDraggedCenterScale(evt);
        if (this.audioPlayer.paused) {
            requestAnimationFrame(this.render.bind(this));
        }
        return false;
    }

    /**
     * Update the center/scale based on a drag event
     * This assumes that scale, center, and centervec have all
     * been defined
     * @param {MouseEvent} evt 
     */
    clickerDraggedCenterScale(evt) {
        let mousePos = getMousePos(evt);
        let X = mousePos.X;
        let Y = mousePos.Y;
        let dX = X - this.lastX;
        let dY = Y - this.lastY;
        this.lastX = X;
        this.lastY = Y;
        if (this.dragging) {
            if (this.clickType == "RIGHT") { //Right click
                this.scale *= Math.pow(1.01, -dY); //Want to zoom in as the mouse goes up
            }
            else if (this.clickType == "LEFT") {
                this.centervec[0] -= 2.0*dX/(this.scale*this.glcanvas.width);
                this.centervec[1] += 2.0*dY/(this.scale*this.glcanvas.height);
            }
        }
    }


}