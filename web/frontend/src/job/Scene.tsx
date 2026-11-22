import { Html, Line, OrbitControls } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { forwardRef, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import type { CameraLink } from './cameraLink'
import { bondRadius, focusMask, fogRange, pickOnChain, stretch } from './chain'
import { scoped } from '../lib/css'
import sheet from './job.module.css'

const cn = scoped(sheet)

const BACKGROUND = '#FFFFFF'
/** The opacity of the chain outside the part in focus. */
const GHOST = 0.2
/** The colour of the hovered bead, one no mark uses. */
const HOVER = '#2A78D6'
/**
 * Takes a mark drawn over the chain out of the pointer events. The pointer would hit the mark
 * instead of the bead under it and the hover would flicker between the two.
 */
const noRaycast = () => null
/**
 * The labels over the canvas take no pointer events. drei's `Html` applies its `pointerEvents` prop
 * only to transformed labels. A label that took the pointer would clear the hover it belongs to.
 */
const MARK_STYLE = { pointerEvents: 'none' } as const
/**
 * The labels are under the hover box and the key. drei gives a label a z-index by its depth, up to
 * millions.
 */
const MARK_LAYER: [number, number] = [1, 0]
/**
 * Hover and pick by the screen. Every bond is projected as its drawn tube and the front tube under
 * the cursor gives the bead. No hover while a drag rotates the view. A click is a press and release
 * that hardly moved.
 */
function Picker({
  points,
  tube,
  onHover,
  onPick,
}: {
  points: [number, number, number][]
  tube: number
  onHover?: (index: number | null, x: number, y: number) => void
  onPick?: (index: number) => void
}) {
  const { camera, gl } = useThree()
  const handlers = useRef({ onHover, onPick })
  handlers.current = { onHover, onPick }
  useEffect(() => {
    const el = gl.domElement
    const v = new THREE.Vector3()
    let frame = 0
    let last: number | null = null
    let down: [number, number] | null = null
    const at = (e: PointerEvent): [number, number] => {
      const r = el.getBoundingClientRect()
      return [e.clientX - r.left, e.clientY - r.top]
    }
    const pick = (x: number, y: number) => {
      const r = el.getBoundingClientRect()
      const cam = camera as THREE.PerspectiveCamera
      const focal = r.height / 2 / Math.tan((cam.fov * Math.PI) / 360)
      const chain = points.map(([px, py, pz]): [number, number, number, number] => {
        v.set(px, py, pz).applyMatrix4(cam.matrixWorldInverse)
        const dist = -v.z
        v.set(px, py, pz).project(cam)
        return [((v.x + 1) / 2) * r.width, ((1 - v.y) / 2) * r.height, dist, dist > 0 ? (tube * focal) / dist : 0]
      })
      return pickOnChain(chain, x, y)
    }
    const move = (e: PointerEvent) => {
      if (e.buttons) return
      const [x, y] = at(e)
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        const i = pick(x, y)
        if (i === null && last === null) return
        last = i
        handlers.current.onHover?.(i, x, y)
      })
    }
    const leave = () => {
      cancelAnimationFrame(frame)
      if (last !== null) handlers.current.onHover?.(null, 0, 0)
      last = null
    }
    const press = (e: PointerEvent) => {
      down = at(e)
    }
    const release = (e: PointerEvent) => {
      if (!down) return
      const [x, y] = at(e)
      const still = Math.hypot(x - down[0], y - down[1]) < 5
      down = null
      if (!still) return
      const i = pick(x, y)
      if (i !== null) handlers.current.onPick?.(i)
    }
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerleave', leave)
    el.addEventListener('pointerdown', press)
    el.addEventListener('pointerup', release)
    return () => {
      cancelAnimationFrame(frame)
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerleave', leave)
      el.removeEventListener('pointerdown', press)
      el.removeEventListener('pointerup', release)
    }
  }, [points, tube, camera, gl])
  return null
}

export type Marker = { label: string; at: [number, number, number] }
const NO_FOCUS: [number, number][] = []
/** `png` gives the view as a picture with its marks drawn on it and `caption` in its lower left corner. */
export type SceneHandle = { reset: () => void; png: (caption?: string) => string | null }

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

/** A light at the camera. It lights the side of a strand that faces the viewer in every direction. */
function CameraLight() {
  const light = useRef<THREE.DirectionalLight>(null)
  useFrame(({ camera }) => light.current?.position.copy(camera.position))
  return <directionalLight ref={light} intensity={1.6} />
}

/** Fog from the middle of the structure to its far side, kept there as the camera zooms. */
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

type Matrices = { joints: THREE.Matrix4[]; tubes: THREE.Matrix4[] }
/** An instance drawn at no size is not drawn. */
const HIDDEN = new THREE.Matrix4().makeScale(0, 0, 0)

/** Returns the place of the sphere of every bead and of the tube of every bond. */
function chainMatrices(points: [number, number, number][]): Matrices {
  const q = new THREE.Quaternion()
  const up = new THREE.Vector3(0, 1, 0)
  const one = new THREE.Vector3(1, 1, 1)
  const joints: THREE.Matrix4[] = []
  const tubes: THREE.Matrix4[] = []
  for (let i = 0; i < points.length; i++) {
    const a = new THREE.Vector3(...points[i])
    joints.push(new THREE.Matrix4().compose(a, q.identity(), one))
    if (i + 1 < points.length) {
      const b = new THREE.Vector3(...points[i + 1])
      const dir = b.clone().sub(a)
      const len = dir.length()
      q.setFromUnitVectors(up, len > 0 ? dir.normalize() : up)
      tubes.push(new THREE.Matrix4().compose(a.clone().add(b).multiplyScalar(0.5), q, new THREE.Vector3(1, len, 1)))
    }
  }
  return { joints, tubes }
}

type PartProps = {
  matrices: Matrices
  colors: string[]
  radius: number
  focus: Uint8Array | null
  /** The part outside the focus, drawn see through. Without it, the part inside, drawn solid. */
  ghost?: boolean
  /** A pass of the ghost that writes its depth and no colour. */
  depthOnly?: boolean
}

/** One part of the chain as lit tubes between beads and spheres at the beads, one instanced mesh
 * each. The solid part holds the beads in focus, or all of them with no focus. The ghost part
 * holds the rest. A bond is in focus when both of its beads are. */
function ChainPart({ matrices, colors, radius, focus, ghost = false, depthOnly = false }: PartProps) {
  const tubes = useRef<THREE.InstancedMesh>(null)
  const joints = useRef<THREE.InstancedMesh>(null)
  const n = matrices.joints.length
  useLayoutEffect(() => {
    const c = new THREE.Color()
    const shown = (inFocus: boolean) => (focus ? inFocus !== ghost : !ghost)
    for (let i = 0; i < n; i++) {
      joints.current?.setMatrixAt(i, shown(!focus || focus[i] === 1) ? matrices.joints[i] : HIDDEN)
      joints.current?.setColorAt(i, c.set(colors[i]))
      if (i + 1 < n && tubes.current) {
        tubes.current.setMatrixAt(i, shown(!focus || (focus[i] === 1 && focus[i + 1] === 1)) ? matrices.tubes[i] : HIDDEN)
        tubes.current.setColorAt(i, c.set(colors[i]))
      }
    }
    for (const mesh of [tubes.current, joints.current]) {
      if (!mesh) continue
      mesh.instanceMatrix.needsUpdate = true
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    }
  }, [matrices, colors, focus, ghost, n])
  // The ghost is drawn twice. The first pass writes the depth of its nearest surface and the second
  // blends its colour there. One pass would add up overlapping spheres and strands to blotches.
  const material = depthOnly ? (
    <meshBasicMaterial colorWrite={false} />
  ) : ghost ? (
    <meshStandardMaterial roughness={0.5} metalness={0.05} transparent opacity={GHOST} depthWrite={false} />
  ) : (
    <meshStandardMaterial roughness={0.5} metalness={0.05} />
  )
  const order = depthOnly ? 1 : ghost ? 2 : 0
  return (
    <>
      {n > 1 && (
        <instancedMesh ref={tubes} args={[undefined, undefined, n - 1]} raycast={noRaycast} frustumCulled={false} renderOrder={order}>
          <cylinderGeometry args={[radius, radius, 1, 10, 1, true]} />
          {material}
        </instancedMesh>
      )}
      <instancedMesh ref={joints} args={[undefined, undefined, n]} raycast={noRaycast} frustumCulled={false} renderOrder={order}>
        <sphereGeometry args={[radius, 12, 10]} />
        {material}
      </instancedMesh>
    </>
  )
}

/** The chain. With a part in focus that part is solid and the rest is see through. */
function Chain({ points, colors, radius, focus }: { points: [number, number, number][]; colors: string[]; radius: number; focus: Uint8Array | null }) {
  const matrices = useMemo(() => chainMatrices(points), [points])
  return (
    <>
      <ChainPart matrices={matrices} colors={colors} radius={radius} focus={focus} />
      {focus && <ChainPart matrices={matrices} colors={colors} radius={radius} focus={focus} ghost depthOnly />}
      {focus && <ChainPart matrices={matrices} colors={colors} radius={radius} focus={focus} ghost />}
    </>
  )
}

type Props = {
  points: [number, number, number][]
  colors: string[]
  markers?: Marker[]
  onHover?: (index: number | null, x: number, y: number) => void
  onPick?: (index: number) => void
  link?: CameraLink
  linkId?: string
  hovered?: number | null
  /** Ranges of chain points drawn at full strength while the rest is faded. None fades nothing. */
  focus?: [number, number][]
}

/** The chain as lit tubes in a WebGL canvas with depth fog, orbit controls, markers
 * and the hovered bead marked. */
export const Scene = forwardRef<SceneHandle, Props>(function Scene({ points, colors, markers = [], onHover, onPick, link, linkId = 'scene', hovered = null, focus = NO_FOCUS }, ref) {
  const controls = useRef<OrbitControlsImpl>(null)
  const gl = useRef<THREE.WebGLRenderer | null>(null)
  const { center, radius } = useMemo(() => {
    const box = new THREE.Box3().setFromPoints(points.map((p) => new THREE.Vector3(...p)))
    const sphere = box.getBoundingSphere(new THREE.Sphere())
    return { center: sphere.center, radius: Math.max(sphere.radius, 1) }
  }, [points])
  const shifted = useMemo(() => points.map(([x, y, z]) => [x - center.x, y - center.y, z - center.z] as [number, number, number]), [points, center])
  const tube = useMemo(() => bondRadius(shifted), [shifted])
  const mask = useMemo(() => focusMask(shifted.length, focus), [shifted.length, focus])
  const near = hovered === null || hovered >= shifted.length ? null : stretch(hovered, shifted.length)
  const shift = (p: [number, number, number]) => [p[0] - center.x, p[1] - center.y, p[2] - center.z] as [number, number, number]
  useImperativeHandle(ref, () => ({
    reset: () => controls.current?.reset(),
    png: (caption?: string) => {
      const view = gl.current?.domElement
      const camera = controls.current?.object
      if (!view) return null
      // The marks of the page are drawn over the view. The picture gets them drawn in.
      const out = document.createElement('canvas')
      out.width = view.width
      out.height = view.height
      const ctx = out.getContext('2d')
      if (!ctx || !camera) return view.toDataURL('image/png')
      ctx.drawImage(view, 0, 0)
      const k = view.width / Math.max(view.clientWidth, 1)
      const at = markers.map((m) => {
        const p = new THREE.Vector3(...shift(m.at)).project(camera)
        return [((p.x + 1) / 2) * view.width, ((1 - p.y) / 2) * view.height] as const
      })
      if (at.length === 2) {
        ctx.strokeStyle = '#151514'
        ctx.lineWidth = 1.5 * k
        ctx.beginPath()
        ctx.moveTo(...at[0])
        ctx.lineTo(...at[1])
        ctx.stroke()
      }
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.font = `700 ${11 * k}px sans-serif`
      markers.forEach((m, i) => {
        ctx.beginPath()
        ctx.arc(at[i][0], at[i][1], 10 * k, 0, 2 * Math.PI)
        ctx.fillStyle = '#151514'
        ctx.fill()
        ctx.strokeStyle = '#FFFFFF'
        ctx.lineWidth = 1.5 * k
        ctx.stroke()
        ctx.fillStyle = '#FFFFFF'
        ctx.fillText(m.label, at[i][0], at[i][1] + 0.5 * k)
      })
      if (caption) {
        ctx.font = `600 ${12 * k}px sans-serif`
        ctx.textAlign = 'left'
        const w = ctx.measureText(caption).width + 16 * k
        ctx.fillStyle = '#151514'
        ctx.fillRect(12 * k, view.height - 36 * k, w, 24 * k)
        ctx.fillStyle = '#FFFFFF'
        ctx.fillText(caption, 20 * k, view.height - 24 * k)
      }
      return out.toDataURL('image/png')
    },
  }))
  return (
    <Canvas
      camera={{ position: [0, 0, radius * 2.4], fov: 45, near: radius / 100, far: radius * 20 }}
      gl={{ preserveDrawingBuffer: true, antialias: true }}
      onCreated={(s) => {
        gl.current = s.gl
      }}
    >
      <color attach="background" args={[BACKGROUND]} />
      <ambientLight intensity={0.45} />
      <CameraLight />
      <DepthFog radius={radius} />
      <Chain points={shifted} colors={colors} radius={tube} focus={mask} />
      <Picker points={shifted} tube={tube} onHover={onHover} onPick={onPick} />
      {near && (
        <>
          <Line points={shifted.slice(near[0], near[1] + 1)} color={HOVER} lineWidth={5} transparent opacity={0.85} depthTest={false} raycast={noRaycast} />
          <mesh position={shifted[hovered!]} renderOrder={2} raycast={noRaycast}>
            <sphereGeometry args={[tube * 2.6, 20, 16]} />
            <meshBasicMaterial color={HOVER} depthTest={false} transparent opacity={0.9} fog={false} />
          </mesh>
          <Html position={shifted[hovered!]} center wrapperClass="scene-mark" style={MARK_STYLE} zIndexRange={MARK_LAYER}>
            <span className={cn('hover-ring')} />
          </Html>
        </>
      )}
      {markers.length === 2 && <Line points={[shift(markers[0].at), shift(markers[1].at)]} color="#151514" lineWidth={1.5} raycast={noRaycast} />}
      {markers.map((m) => (
        <Html key={m.label} position={shift(m.at)} center wrapperClass="scene-mark" style={MARK_STYLE} zIndexRange={MARK_LAYER}>
          <span className={cn('marker')}>{m.label}</span>
        </Html>
      ))}
      <OrbitControls ref={controls} makeDefault enableDamping={false} />
      {link && <CameraSync link={link} id={linkId} />}
    </Canvas>
  )
})
