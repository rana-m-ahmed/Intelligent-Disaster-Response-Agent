/**
 * OrbitControls for Three.js
 * Simplified camera control for desktop and mobile.
 */

const OrbitControls = function (camera, domElement) {
    this.camera = camera;
    this.domElement = domElement;

    this.target = new THREE.Vector3();
    this.distance = 25;
    this.phi = Math.PI / 3;
    this.theta = Math.PI / 4;

    this.enableRotate = true;
    this.rotateSpeed = 0.01;

    this.isDragging = false;
    this.previousMousePosition = { x: 0, y: 0 };

    this.domElement.addEventListener("mousedown", this.onMouseDown.bind(this));
    this.domElement.addEventListener("mousemove", this.onMouseMove.bind(this));
    this.domElement.addEventListener("mouseup", this.onMouseUp.bind(this));
    this.domElement.addEventListener("wheel", this.onMouseWheel.bind(this), false);
};

OrbitControls.prototype.onMouseDown = function (event) {
    this.isDragging = true;
    this.previousMousePosition = { x: event.clientX, y: event.clientY };
};

OrbitControls.prototype.onMouseMove = function (event) {
    if (!this.isDragging || !this.enableRotate) return;

    const deltaX = event.clientX - this.previousMousePosition.x;
    const deltaY = event.clientY - this.previousMousePosition.y;

    this.theta += deltaX * this.rotateSpeed;
    this.phi -= deltaY * this.rotateSpeed;
    this.phi = Math.max(0.1, Math.min(Math.PI - 0.1, this.phi));

    this.previousMousePosition = { x: event.clientX, y: event.clientY };
};

OrbitControls.prototype.onMouseUp = function () {
    this.isDragging = false;
};

OrbitControls.prototype.onMouseWheel = function (event) {
    event.preventDefault();
    this.distance += event.deltaY > 0 ? 1 : -1;
    this.distance = Math.max(5, Math.min(100, this.distance));
};

OrbitControls.prototype.update = function () {
    const x = this.target.x + this.distance * Math.sin(this.phi) * Math.cos(this.theta);
    const y = this.target.y + this.distance * Math.cos(this.phi);
    const z = this.target.z + this.distance * Math.sin(this.phi) * Math.sin(this.theta);

    this.camera.position.set(x, y, z);
    this.camera.lookAt(this.target);
};
