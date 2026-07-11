import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { useEffect, useRef } from "react";
import * as THREE from "three";

import type { ScenePreview } from "./types";

interface Props {
  scene: ScenePreview;
  selected: string | null;
  onSelect: (id: string) => void;
}

const materialColor = (material: string) =>
  material === "pec" ? 0xd09a52 : material.includes("fr4") ? 0x3f936c : 0x7899a8;

export function SceneViewer({ scene, selected, onSelect }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  useEffect(() => {
    if (!host.current) return;
    const element = host.current;
    const world = new THREE.Scene();
    world.background = new THREE.Color(0x07100d);
    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
    camera.position.set(1.2, 1.1, 1.25);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    element.appendChild(renderer.domElement);
    renderer.domElement.setAttribute("aria-label", "Interactive 3D experiment geometry");
    renderer.domElement.setAttribute("role", "img");

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);

    world.add(new THREE.AmbientLight(0xffffff, 1.5));
    const key = new THREE.DirectionalLight(0xffe2aa, 2.2);
    key.position.set(2, 3, 4);
    world.add(key);
    const fill = new THREE.DirectionalLight(0x5cbca0, 1.1);
    fill.position.set(-3, -1, 2);
    world.add(fill);

    const scale = 1 / Math.max(...scene.domain_m);
    const selectable: THREE.Mesh[] = [];
    for (const entity of scene.entities) {
      const [lo, hi] = entity.bounds_m;
      const size = hi.map((value, index) => Math.max((value - lo[index]) * scale, 0.018));
      const center = hi.map((value, index) => ((value + lo[index]) / 2 - scene.domain_m[index] / 2) * scale);
      const geometry = new THREE.BoxGeometry(size[0], size[2], size[1]);
      const isSelected = selected === entity.id;
      const material = new THREE.MeshStandardMaterial({
        color: isSelected ? 0xf6cf78 : materialColor(entity.material_id),
        metalness: entity.material_id === "pec" ? 0.72 : 0.05,
        roughness: entity.material_id === "pec" ? 0.28 : 0.7,
        transparent: entity.material_id !== "pec",
        opacity: entity.material_id !== "pec" ? 0.72 : 1,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(center[0], center[2], center[1]);
      mesh.userData.id = entity.id;
      world.add(mesh);
      selectable.push(mesh);
      if (isSelected) {
        const edge = new THREE.LineSegments(
          new THREE.EdgesGeometry(geometry),
          new THREE.LineBasicMaterial({ color: 0xffe7a7 }),
        );
        edge.position.copy(mesh.position);
        world.add(edge);
      }
    }

    for (const overlay of scene.overlays) {
      const color = overlay.role === "excitation" ? 0xf25f5c : 0x62d2ff;
      if (overlay.position_m) {
        const position = overlay.position_m.map(
          (value, index) => (value - scene.domain_m[index] / 2) * scale,
        );
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(0.028, 18, 18),
          new THREE.MeshBasicMaterial({ color }),
        );
        marker.position.set(position[0], position[2], position[1]);
        world.add(marker);
      } else if (overlay.coordinate_m !== undefined && overlay.axis) {
        const planeSize = scene.domain_m.map((value) => value * scale * 0.88);
        const geometry = new THREE.PlaneGeometry(planeSize[0], planeSize[1]);
        const plane = new THREE.Mesh(
          geometry,
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.15, side: THREE.DoubleSide }),
        );
        const normalized = (overlay.coordinate_m - scene.domain_m[2] / 2) * scale;
        plane.position.z = normalized;
        world.add(plane);
      }
    }

    const domain = new THREE.BoxGeometry(
      scene.domain_m[0] * scale,
      scene.domain_m[2] * scale,
      scene.domain_m[1] * scale,
    );
    world.add(
      new THREE.LineSegments(
        new THREE.EdgesGeometry(domain),
        new THREE.LineBasicMaterial({ color: 0x356050, transparent: true, opacity: 0.55 }),
      ),
    );
    world.add(new THREE.GridHelper(1.25, 10, 0x24483b, 0x142820));

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const click = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(selectable)[0];
      if (hit?.object.userData.id) onSelectRef.current(hit.object.userData.id);
    };
    renderer.domElement.addEventListener("pointerdown", click);

    const resize = () => {
      const width = Math.max(element.clientWidth, 1);
      const height = Math.max(element.clientHeight, 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    resize();
    let frame = 0;
    const animate = () => {
      controls.update();
      renderer.render(world, camera);
      frame = requestAnimationFrame(animate);
    };
    animate();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", click);
      controls.dispose();
      renderer.dispose();
      world.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => material.dispose());
        }
      });
      element.replaceChildren();
    };
  }, [scene, selected]);

  return <div ref={host} className="scene-canvas" data-testid="scene-viewer" />;
}
