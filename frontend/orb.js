/**
 * NutriAssist — 3D Orb (Three.js)
 * Interactive animated orb that responds to system state.
 */

class NutriOrb {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.state = 'idle'; // idle | listening | processing | eat | avoid | sometimes
        this.targetColor = { r: 0.937, g: 0.624, b: 0.153 }; // inner core
        this.currentColor = { ...this.targetColor };
        this.time = 0;
        this.noiseScale = 0.3;
        this.targetNoiseScale = 0.3;

        this._init();
        this._animate();
    }

    _init() {
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;

        // Scene
        this.scene = new THREE.Scene();

        // Camera
        this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100);
        this.camera.position.z = 4;

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true,
        });
        this.renderer.setSize(w, h);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setClearColor(0x000000, 0);

        // Geometry — icosahedron for organic look
        const geometry = new THREE.IcosahedronGeometry(1.2, 6);
        this._basePositions = geometry.attributes.position.array.slice();

        // Material — custom shader
        this.material = new THREE.ShaderMaterial({
            uniforms: {
                uTime: { value: 0 },
                uColor1: { value: new THREE.Color(0.937, 0.624, 0.153) }, // #EF9F27 highlight center
                uColor2: { value: new THREE.Color(0.522, 0.310, 0.043) }, // #854F0B accent mid gradient
                uNoiseScale: { value: 0.3 },
                uGlowIntensity: { value: 0.6 },
            },
            vertexShader: `
                uniform float uTime;
                uniform float uNoiseScale;
                varying vec3 vNormal;
                varying vec3 vPosition;
                varying float vDisplacement;

                // Simplex-ish noise
                vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
                vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
                vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
                vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

                float snoise(vec3 v) {
                    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
                    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
                    vec3 i  = floor(v + dot(v, C.yyy));
                    vec3 x0 = v - i + dot(i, C.xxx);
                    vec3 g = step(x0.yzx, x0.xyz);
                    vec3 l = 1.0 - g;
                    vec3 i1 = min(g.xyz, l.zxy);
                    vec3 i2 = max(g.xyz, l.zxy);
                    vec3 x1 = x0 - i1 + C.xxx;
                    vec3 x2 = x0 - i2 + C.yyy;
                    vec3 x3 = x0 - D.yyy;
                    i = mod289(i);
                    vec4 p = permute(permute(permute(
                        i.z + vec4(0.0, i1.z, i2.z, 1.0))
                        + i.y + vec4(0.0, i1.y, i2.y, 1.0))
                        + i.x + vec4(0.0, i1.x, i2.x, 1.0));
                    float n_ = 0.142857142857;
                    vec3 ns = n_ * D.wyz - D.xzx;
                    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
                    vec4 x_ = floor(j * ns.z);
                    vec4 y_ = floor(j - 7.0 * x_);
                    vec4 x = x_ * ns.x + ns.yyyy;
                    vec4 y = y_ * ns.x + ns.yyyy;
                    vec4 h = 1.0 - abs(x) - abs(y);
                    vec4 b0 = vec4(x.xy, y.xy);
                    vec4 b1 = vec4(x.zw, y.zw);
                    vec4 s0 = floor(b0)*2.0 + 1.0;
                    vec4 s1 = floor(b1)*2.0 + 1.0;
                    vec4 sh = -step(h, vec4(0.0));
                    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
                    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
                    vec3 p0 = vec3(a0.xy, h.x);
                    vec3 p1 = vec3(a0.zw, h.y);
                    vec3 p2 = vec3(a1.xy, h.z);
                    vec3 p3 = vec3(a1.zw, h.w);
                    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
                    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
                    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
                    m = m * m;
                    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
                }

                void main() {
                    vNormal = normalize(normalMatrix * normal);
                    vec3 pos = position;

                    float noise = snoise(pos * 1.5 + uTime * 0.4) * uNoiseScale;
                    float noise2 = snoise(pos * 3.0 + uTime * 0.2) * uNoiseScale * 0.3;
                    vDisplacement = noise + noise2;

                    pos += normal * (noise + noise2);
                    vPosition = pos;

                    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 uColor1;
                uniform vec3 uColor2;
                uniform float uGlowIntensity;
                uniform float uTime;
                varying vec3 vNormal;
                varying vec3 vPosition;
                varying float vDisplacement;

                void main() {
                    vec3 outerColor = vec3(0.522, 0.310, 0.043); // #854F0B
                    vec3 fadeColor = vec3(0.969, 0.929, 0.847);  // #F7EDD8

                    // Fresnel edge glow
                    vec3 viewDir = normalize(cameraPosition - vPosition);
                    float fresnel = pow(1.0 - max(dot(viewDir, vNormal), 0.0), 2.5);

                    // Color mix based on displacement
                    float mixFactor = clamp(vDisplacement * 1.6 + 0.5, 0.0, 1.0);
                    vec3 coreToMid = mix(uColor1, uColor2, mixFactor);
                    vec3 baseColor = mix(coreToMid, outerColor, clamp(fresnel * 0.75, 0.0, 1.0));

                    // Add subtle animation color shift
                    float shift = sin(uTime * 0.5) * 0.02;
                    baseColor += shift;

                    // Rim glow
                    vec3 glowColor = vec3(0.937, 0.624, 0.153);
                    vec3 finalColor = baseColor + fresnel * glowColor * (uGlowIntensity * 0.28);

                    // Inner ambient
                    float ambient = 0.10;
                    finalColor += ambient * baseColor;

                    // Fade toward background tone at edges
                    finalColor = mix(finalColor, fadeColor, clamp(fresnel * 0.35, 0.0, 1.0));

                    gl_FragColor = vec4(finalColor, 0.90 - fresnel * 0.14);
                }
            `,
            transparent: true,
            side: THREE.FrontSide,
        });

        this.mesh = new THREE.Mesh(geometry, this.material);
        this.scene.add(this.mesh);

        // Handle resize
        window.addEventListener('resize', () => this._onResize());
    }

    _onResize() {
        const w = this.canvas.clientWidth;
        const h = this.canvas.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    _animate() {
        requestAnimationFrame(() => this._animate());

        this.time += 0.016;
        this.material.uniforms.uTime.value = this.time;

        // Smooth color lerp
        const lerpSpeed = 0.03;
        this.currentColor.r += (this.targetColor.r - this.currentColor.r) * lerpSpeed;
        this.currentColor.g += (this.targetColor.g - this.currentColor.g) * lerpSpeed;
        this.currentColor.b += (this.targetColor.b - this.currentColor.b) * lerpSpeed;

        this.material.uniforms.uColor1.value.setRGB(
            this.currentColor.r, this.currentColor.g, this.currentColor.b
        );
        this.material.uniforms.uColor2.value.setRGB(
            0.522,
            0.310,
            0.043
        );

        // Smooth noise scale
        this.noiseScale += (this.targetNoiseScale - this.noiseScale) * 0.02;
        this.material.uniforms.uNoiseScale.value = this.noiseScale;

        // Gentle rotation
        this.mesh.rotation.y += 0.003;
        this.mesh.rotation.x += 0.001;

        this.renderer.render(this.scene, this.camera);
    }

    setState(state) {
        this.state = state;
        const statusEl = document.getElementById('orb-status');
        const statusText = statusEl.querySelector('.status-text');

        statusEl.className = 'orb-status ' + state;

        switch (state) {
            case 'idle':
                this.targetColor = { r: 0.937, g: 0.624, b: 0.153 };
                this.targetNoiseScale = 0.3;
                this.material.uniforms.uGlowIntensity.value = 0.6;
                statusText.textContent = 'Ready to assist';
                break;

            case 'listening':
                this.targetColor = { r: 0.937, g: 0.624, b: 0.153 };
                this.targetNoiseScale = 0.6;
                this.material.uniforms.uGlowIntensity.value = 0.8;
                statusText.textContent = 'Listening...';
                break;

            case 'processing':
                this.targetColor = { r: 0.522, g: 0.310, b: 0.043 };
                this.targetNoiseScale = 0.8;
                this.material.uniforms.uGlowIntensity.value = 0.75;
                statusText.textContent = 'Analyzing...';
                break;

            case 'eat':
                this.targetColor = { r: 0.937, g: 0.624, b: 0.153 };
                this.targetNoiseScale = 0.25;
                this.material.uniforms.uGlowIntensity.value = 0.62;
                statusText.textContent = 'Good choice!';
                break;

            case 'avoid':
                this.targetColor = { r: 0.937, g: 0.624, b: 0.153 };
                this.targetNoiseScale = 0.5;
                this.material.uniforms.uGlowIntensity.value = 0.68;
                statusText.textContent = 'Not recommended';
                break;

            case 'sometimes':
                this.targetColor = { r: 0.522, g: 0.310, b: 0.043 };
                this.targetNoiseScale = 0.35;
                this.material.uniforms.uGlowIntensity.value = 0.58;
                statusText.textContent = 'In moderation';
                break;
        }
    }
}

// Initialize on load
let nutriOrb;
document.addEventListener('DOMContentLoaded', () => {
    nutriOrb = new NutriOrb('orb-canvas');
});
