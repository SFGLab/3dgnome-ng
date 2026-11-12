import { Html, Line, OrbitControls } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import type { ThreeEvent } from '@react-three/fiber'
import { forwardRef, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import type { CameraLink } from './cameraLink'
import { bondRadius, fogRange, stretch } from './chain'

const BACKGROUND = '#FFFFFF'
/** The hovered bead's colour, one no mark uses. */
const HOVER = '#2A78D6'
/** Marks drawn over the chain take no pointer events, or the pointer would hit the mark instead
 * of the bead under it and the hover would flicker between the two. */
const noRaycast = () => null

export type Marker = { label: string; at: [number, number, number] }
export type SceneHandle = { reset: () => void; png: () => string | null }


function CameraSync({ link, id }: { link: CameraLink; id: string }) {
  const { camera, controls } = useThree() as unknown as { camera: THREE.PerspectiveCamera; controls: OrbitControlsImpl | null }
  useEffect(() => {
    if (!controls) return
    let applying = false
    const onChange = () => {
      if (!applying) link.publish(id, { position: camera.position.toArray(), target: controls.target.toArray(), zoom: camera.zoom })
    }
    controls.addEventListener('change', onChange)
    const off = link.subscribe(id, (s) => {
      applying = true
      camera.position.fromArray(s.position)
      controls.target.fromArray(s.target)
      controls.update()
      applying = false
    })
    return () => {
      controls.removeEventListener('change', onChange)
      off()
    }
  }, [camera, controls, link, id])
  return null
}

/** A light at the camera, so the side of a strand facing the viewer is lit wherever it turns. */
function CameraLight() {
  const light = useRef<THREE.DirectionalLight>(null)
  useFrame(({ camera }) => light.current?.position.copy(camera.position))
  return <directionalLight ref={light} intensity={1.6} />
}

/** Fog from the structure's middle to its far side, kept there as the camera zooms. */
function DepthFog({ radius }: { radius: number }) {
  const { scene } = useThree()
  useEffect(() => {
    scene.fog = new THREE.Fog(BACKGROUND, 1, 2)
    return () => {
      scene.fog = null
    }
  }, [scene])
  useFrame(({ camera }) => {
    const fog = scene.fog as THREE.Fog | null
    if (!fog) return
    const [near, far] = fogRange(radius, camera.position.length())
    fog.near = near
    fog.far = far
  })
  return null
}

type ChainProps = {
  points: [number, number, number][]
  colors: string[]
  radius: number
  onHover?: (index: number | null, x: number, y: number) => void
  onPick?: (index: number) => void
}

/** The chain as lit tubes between beads and spheres at the beads, one instanced mesh each. */
function Chain({ points, colors, radius, onHover, onPick }: ChainProps) {
  const tubes = useRef<THREE.InstancedMesh>(null)
  const joints = useRef<THREE.InstancedMesh>(null)
  const n = points.length
  useLayoutEffect(() => {
    const m = new THREE.Matrix4()
    const q = new THREE.Quaternion()
    const up = new THREE.Vector3(0, 1, 0)
    const one = new THREE.Vector3(1, 1, 1)
    const c = new THREE.Color()
    for (let i = 0; i < n; i++) {
      const a = new THREE.Vector3(...points[i])
      joints.current?.setMatrixAt(i, m.compose(a, q.identity(), one))
      joints.current?.setColorAt(i, c.set(colors[i]))
      if (i + 1 < n && tubes.current) {
        const b = new THREE.Vector3(...points[i + 1])
        const dir = b.clone().sub(a)
        const len = dir.length()
        q.setFromUnitVectors(up, len > 0 ? dir.normalize() : up)
        tubes.current.setMatrixAt(i, m.compose(a.clone().add(b).multiplyScalar(0.5), q, new THREE.Vector3(1, len, 1)))
        tubes.current.setColorAt(i, c.set(colors[i]))
      }
    }
    for (const mesh of [tubes.current, joints.current]) {
      if (!mesh) continue
      mesh.instanceMatrix.needsUpdate = true
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
      mesh.computeBoundingSphere()
    }
  }, [points, colors, n])
  const handlers = (bead: (id: number) => number) => ({
    onPointerMove: (e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation()
      if (e.instanceId !== undefined) onHover?.(bead(e.instanceId), e.nativeEvent.offsetX, e.nativeEvent.offsetY)
    },
    onPointerOut: () => onHover?.(null, 0, 0),
    onClick: (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation()
      if (e.instanceId !== undefined) onPick?.(bead(e.instanceId))
    },
  })
  return (
    <>
      {n > 1 && (
        <instancedMesh ref={tubes} args={[undefined, undefined, n - 1]} {...handlers((id) => id)}>
          <cylinderGeometry args={[radius, radius, 1, 10, 1, true]} />
          <meshStandardMaterial roughness={0.5} metalness={0.05} />
        </instancedMesh>
      )}
      <instancedMesh ref={joints} args={[undefined, undefined, n]} {...handlers((id) => id)}>
        <sphereGeometry args={[radius, 12, 10]} />
        <meshStandardMaterial roughness={0.5} metalness={0.05} />
      </instancedMesh>
    </>
  )
}

type Props = {
  points: [number, number, number][]
  colors: string[]
  markers?: Marker[]
  highlight?: [number, number, number][][]
  onHover?: (index: number | null, x: number, y: number) => void
  onPick?: (index: number) => void
  link?: CameraLink
  linkId?: string
  hovered?: number | null
}

/** The chain as lit tubes in a WebGL canvas with depth fog, orbit controls, markers, highlights
 * and the hovered bead marked. */
export const Scene = forwardRef<SceneHandle, Props>(function Scene({ points, colors, markers = [], highlight = [], onHover, onPick, link, linkId = 'scene', hovered = null }, ref) {
  const controls = useRef<OrbitControlsImpl>(null)
  const gl = useRef<THREE.WebGLRenderer | null>(null)
  const { center, radius } = useMemo(() => {
    const box = new THREE.Box3().setFromPoints(points.map((p) => new THREE.Vector3(...p)))
    const sphere = box.getBoundingSphere(new THREE.Sphere())
    return { center: sphere.center, radius: Math.max(sphere.radius, 1) }
  }, [points])
  const shifted = useMemo(() => points.map(([x, y, z]) => [x - center.x, y - center.y, z - center.z] as [number, number, number]), [points, center])
  const tube = useMemo(() => bondRadius(shifted), [shifted])
  const near = hovered === null || hovered >= shifted.length ? null : stretch(hovered, shifted.length)
  useImperativeHandle(ref, () => ({
    reset: () => controls.current?.reset(),
    png: () => gl.current?.domElement.toDataURL('image/png') ?? null,
  }))
  const shift = (p: [number, number, number]) => [p[0] - center.x, p[1] - center.y, p[2] - center.z] as [number, number, number]
  return (
    <Canvas
      camera={{ position: [0, 0, radius * 2.4], fov: 45, near: radius / 100, far: radius * 20 }}
      gl={{ preserveDrawingBuffer: true, antialias: true }}
      onCreated={(s) => {
        gl.current = s.gl
      }}
      onPointerMissed={() => onHover?.(null, 0, 0)}
    >
      <color attach="background" args={[BACKGROUND]} />
      <ambientLight intensity={0.55} />
      <CameraLight />
      <DepthFog radius={radius} />
      <Chain points={shifted} colors={colors} radius={tube} onHover={onHover} onPick={onPick} />
      {near && (
        <>
          <Line points={shifted.slice(near[0], near[1] + 1)} color={HOVER} lineWidth={5} transparent opacity={0.85} depthTest={false} raycast={noRaycast} />
          <mesh position={shifted[hovered!]} renderOrder={2} raycast={noRaycast}>
            <sphereGeometry args={[tube * 2.6, 20, 16]} />
            <meshBasicMaterial color={HOVER} depthTest={false} transparent opacity={0.9} fog={false} />
          </mesh>
          <Html position={shifted[hovered!]} center>
            <span className="hover-ring" />
          </Html>
        </>
      )}
      {highlight.map((seg, i) => (
        <Line key={i} points={seg.map(shift)} color="#151514" lineWidth={7} transparent opacity={0.25} raycast={noRaycast} />
      ))}
      {markers.length === 2 && <Line points={[shift(markers[0].at), shift(markers[1].at)]} color="#151514" lineWidth={1.5} raycast={noRaycast} />}
      {markers.map((m) => (
        <Html key={m.label} position={shift(m.at)} center>
          <span className="marker">{m.label}</span>
        </Html>
      ))}
      <OrbitControls ref={controls} makeDefault enableDamping={false} />
      {link && <CameraSync link={link} id={linkId} />}
    </Canvas>
  )
})
