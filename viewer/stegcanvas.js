let DSP_WORKER_PATH = "dspworker.js";

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



const VERT_SRC_3D = `
attribute vec4 a_info;
attribute vec3 a_color;

// Uniforms set from Javascript that are constant
// over all fragments
uniform mat4 uMVMatrix; // Where the origin (0, 0) is on the canvas
uniform mat4 uPMatrix;
uniform float uPointSize; // Size to draw points

varying float v_time;
varying vec3 v_color;

void main() {
  gl_PointSize = uPointSize;
  vec4 a_position = vec4(a_info.x, a_info.y, a_info.z, 1.0);
  gl_Position = uPMatrix * uMVMatrix * a_position;
  v_time = a_info.w;
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

const mat4 = glMatrix.mat4;
const vec3 = glMatrix.vec3;


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

class StegCanvas {
    /**
     * 
     * @param {DOM Element} glcanvas Handle to GL Canvas
     * @param {DOM Element} audioPlayer Handle to audio player object
     * @param {Float32Array} audioSamples Audio samples
     * @param {int} sr Sample rate
     * @param {Dat.GUI Folder} folder Handle to dat.gui folder for options
     * @param {object} params Default parameters to use
     */
    constructor(glcanvas, audioPlayer, audioSamples, sr, folder, params) {
        this.audioPlayer = audioPlayer;
        this.audioSamples = audioSamples;
        this.sr = sr;
        const that = this;
        glcanvas.addEventListener("contextmenu", function(e){ e.stopPropagation(); e.preventDefault(); return false; }); //Need this to disable the menu that pops up on right clicking
        this.colorBuffers = {};

        // 2D Viewing options
        this.centervec = [0, 0];
        this.scale = 1;

        // 3D Viewing options
        this.mvMatrix = mat4.create();
        this.pMatrix = mat4.create();
        this.camera = new MousePolarCamera(glcanvas.width, glcanvas.height, 0.75);
        this.farR = 1.0;
        this.bbox = [0, 1, 0, 1, 0, 1];

        // Decoder parameters
        this.coords = [];
        if (params === undefined) {
            params = {};
        }
        let defaultParams = {"win":1024, "shift":0, "L":16, "Q":4, "freq1":1, "freq2":2, "freq3":-1, "colormap":"Spectral"};
        for (let param in defaultParams) {
            if (param in params) {
                this[param] = params[param];
            }
            else {
                this[param] = defaultParams[param];
            }
        }
        let colormap = this.colormap;
        
        try {
            glcanvas.gl = glcanvas.getContext("webgl");
            glcanvas.gl.viewportWidth = glcanvas.width;
            glcanvas.gl.viewportHeight = glcanvas.height;
            this.glcanvas = glcanvas;
            this.setupMenus(folder, colormap);
            this.loadShaders();
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
        this.winGUI = folder.add(this, "win", 0, this.sr, 1).onChange(this.extractSignal.bind(this));
        this.shiftGUI = folder.add(this, "shift", 0, this.win, 1).listen().onChange(this.extractSignal.bind(this));
        this.winGUI.onChange(
            function(v) {
                that.shiftGUI.max(v);
                that.shift = 0;
            }
        );
        folder.add(this, "L", 0).onChange(this.extractSignal.bind(this));
        folder.add(this, "Q", 1, 10, 1).onChange(this.extractSignal.bind(this));
        folder.add(this, "freq1", 0, Math.floor(this.win/2), 1).onChange(this.extractSignal.bind(this));
        folder.add(this, "freq2", 0, Math.floor(this.win/2), 1).onChange(this.extractSignal.bind(this));
        folder.add(this, "freq3", -1, Math.floor(this.win/2), 1).onChange(this.extractSignal.bind(this));
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
     * Load in the shaders for 2D and 3D drawing
     */
    loadShaders() {
        let gl = this.glcanvas.gl;

        // Step 1: Setup 2D Shader
        let shader2d = getShaderProgram(gl, "stipple", VERT_SRC, FRAG_SRC);
        // Extract uniforms and store them in the shader object
        shader2d.uCenterUniform = gl.getUniformLocation(shader2d, "uCenter");
        shader2d.uScaleUniform = gl.getUniformLocation(shader2d, "uScale");
        shader2d.uTimeUniform = gl.getUniformLocation(shader2d, "uTime");
        shader2d.uPointSizeUniform = gl.getUniformLocation(shader2d, "uPointSize");
        // Extract the info buffer and store it in the shader object
        shader2d.infoLocation = gl.getAttribLocation(shader2d, "a_info");
        gl.enableVertexAttribArray(shader2d.infoLocation);
        // Extract the colormap buffer and store it in the shader object
        shader2d.colorLocation = gl.getAttribLocation(shader2d, "a_color");
        gl.enableVertexAttribArray(shader2d.colorLocation);
        this.shader2d = shader2d;

        // Step 2: Setup 3D Shader
        let shader3d = getShaderProgram(gl, "stipple", VERT_SRC_3D, FRAG_SRC);
        // Extract uniforms and store them in the shader object
        shader3d.uPMatrixUniform = gl.getUniformLocation(shader3d, "uPMatrix");
        shader3d.uMVMatrixUniform = gl.getUniformLocation(shader3d, "uMVMatrix");
        shader3d.uTimeUniform = gl.getUniformLocation(shader3d, "uTime");
        shader3d.uPointSizeUniform = gl.getUniformLocation(shader3d, "uPointSize");
        // Extract the info buffer and store it in the shader object
        shader3d.infoLocation = gl.getAttribLocation(shader3d, "a_info");
        gl.enableVertexAttribArray(shader3d.infoLocation);
        // Extract the colormap buffer and store it in the shader object
        shader3d.colorLocation = gl.getAttribLocation(shader3d, "a_color");
        gl.enableVertexAttribArray(shader3d.colorLocation);
        this.shader3d = shader3d;

        // Step 3: Create info buffer
        this.infoBuffer = gl.createBuffer();
    }

    /**
     * Extract the hidden signal and fill in the coordinates
     */
    extractSignal() {
        const that = this;
        const progressBar = this.progressBar;
        new Promise((resolve, reject) => {
            const worker = new Worker(DSP_WORKER_PATH);
            let freqs = [this.freq1, this.freq2];
            if (this.freq3 > -1) {
                freqs.push(this.freq3);
            }
            let payload = {samples:that.audioSamples, win:that.win, shift:that.shift, L:that.L, Q:that.Q, freqs:freqs};
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
                    that.coords = event.data.coords;
                    that.colorBuffers = {};
                    that.setupBuffers();
                    resolve();
                }
                else {
                    console.log("Unknown command");
                }
            }
        }).then(() => {
            progressBar.changeToReady("Ready!  Press play!");
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
        const dim = this.coords.length+1;
        const N = this.coords[0].length;
        let info = new Float32Array(N*(dim+1));
        // Finish setting up buffers
        for (let i = 0; i < N; i++) {
            for (let k = 0; k < dim-1; k++) {
                info[i*dim+k] = this.coords[k][i];
            }
            info[(i+1)*dim-1] = i/N;
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.infoBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, info, gl.STATIC_DRAW);
        let shader = this.shader2d;
        if (dim > 3) {
            shader = this.shader3d;
            this.centerOnBBox();
        }
        gl.vertexAttribPointer(shader.infoLocation, dim, gl.FLOAT, false, 0, 0);
        this.render();
    }

    /**
     * Center 3D camera on the bounding box of the curve
     */
    centerOnBBox() {
        const coords = this.coords;
        this.bbox = [coords[0][0], coords[0][0], 
                     coords[1][0], coords[1][0],
                     coords[2][0], coords[2][0]];
        for (let i = 0; i < this.coords[0].length; i++) {
            if (coords[0][i] < this.bbox[0]) {
                this.bbox[0] = coords[0][i];
            }
            if (coords[0][i] > this.bbox[1]) {
                this.bbox[1] = coords[0][i];
            }
            if (coords[1][i] < this.bbox[2]) {
                this.bbox[2] = coords[1][i];
            }
            if (coords[1][i] > this.bbox[3]) {
                this.bbox[3] = coords[1][i];
            }
            if (coords[2][i] < this.bbox[4]) {
                this.bbox[4] = coords[2][i];
            }
            if (coords[2][i] > this.bbox[4]) {
                this.bbox[5] = coords[2][i];
            }
        }
        var dX = this.bbox[1] - this.bbox[0];
        var dY = this.bbox[3] - this.bbox[2];
        var dZ = this.bbox[5] - this.bbox[4];
        this.farR = Math.sqrt(dX*dX + dY*dY + dZ*dZ);
        this.camera.R = this.farR;
        this.camera.center = vec3.fromValues(this.bbox[0] + 0.5*dX, this.bbox[2] + 0.5*dY, this.bbox[4] + 0.5*dZ);
        this.camera.phi = Math.PI/2;
        this.camera.theta = -Math.PI/2;
        this.camera.updateVecsFromPolar();
    };

    /**
     * Setup the buffer for a particular colormap
     * @param {string} key Colormap to use
     */
    setupColorBuffer(key) {
        const N = this.coords[0].length;
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

        // Step 1: Setup uniform variables that are sent to the shaders
        let shader = null;
        let dim = 3;
        if (this.coords.length == 2) {
            shader = this.shader2d;
            gl.useProgram(shader);
            gl.uniform2fv(shader.uCenterUniform, this.centervec);
            gl.uniform1f(shader.uScaleUniform, this.scale);
        }
        else if (this.coords.length > 2) {
            shader = this.shader3d;
            dim = 4;
            gl.useProgram(shader);
            mat4.perspective(this.pMatrix, 45, gl.viewportWidth / gl.viewportHeight, this.camera.R/100.0, Math.max(this.farR*2, this.camera.R*2));
            this.mvMatrix = this.camera.getMVMatrix();
            gl.uniformMatrix4fv(shader.uPMatrixUniform, false, this.pMatrix);
            gl.uniformMatrix4fv(shader.uMVMatrixUniform, false, this.mvMatrix);
        }
        
        
        let t = this.audioPlayer.currentTime / this.audioPlayer.duration;
        gl.uniform1f(shader.uTime, t);

        let N = Math.floor(this.audioPlayer.currentTime*this.sr/this.win);
        N = Math.min(N, this.coords[0].length);

        // Step 2: Bind vertex and time buffers to draw points
        gl.bindBuffer(gl.ARRAY_BUFFER, this.infoBuffer);
        gl.vertexAttribPointer(shader.infoLocation, dim, gl.FLOAT, false, 0, 0);
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
        let mousePos = getMousePos(evt);
        let X = mousePos.X;
        let Y = mousePos.Y;
        let dX = X - this.lastX;
        let dY = Y - this.lastY;
        this.lastX = X;
        this.lastY = Y;
        if (this.coords.length < 3) {
            this.clickerDraggedCenterScale2D(dX, dY);
        }
        else {
            this.clickerDraggedPerspective3D(dX, dY, evt);
        }
        if (this.audioPlayer.paused) {
            requestAnimationFrame(this.render.bind(this));
        }
        return false;
    }

    /**
     * Update the center/scale based on a drag event
     * This assumes that scale, center, and centervec have all
     * been defined
     * @param {int} dX Change in mouse X coordinate
     * @param {int} dY Change in mouse Y coordinate
     */
    clickerDraggedCenterScale2D(dX, dY) {
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

    /**
     * Update the 3D camera based on a drag event
     * @param {int} dX Change in mouse X coordinate
     * @param {int} dY Change in mouse Y coordinate
     * @param {MouseEvent} evt Mouse event
     */
    clickerDraggedPerspective3D(dX, dY, evt) {
        if (this.dragging) {
            if (this.clickType == "MIDDLE") {
                this.camera.translate(dX, -dY);
            }
            else if (this.clickType == "RIGHT") { //Right click
                this.camera.zoom(dY); //Want to zoom in as the mouse goes up
            }
            else if (this.clickType == "LEFT") {
                if (evt.ctrlKey) {
                    this.camera.translate(dX, -dY);
                }
                else if (evt.shiftKey) {
                    this.camera.zoom(dY);
                }
                else{
                    this.camera.orbitLeftRight(dX);
                    this.camera.orbitUpDown(-dY);
                }
            }
        }
    }


}