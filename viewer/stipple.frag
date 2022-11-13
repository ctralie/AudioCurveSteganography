precision highp float;

varying float v_time;
uniform float uTime;

void main() {
    gl_FragColor = vec4(0, 0, v_time, 1);
}
