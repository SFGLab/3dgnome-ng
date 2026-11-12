import { Html, Line, OrbitControls } from '@react-three/drei'
import { Canvas, useThree } from '@react-three/fiber'
import type { ThreeEvent } from '@react-three/fiber'
import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import type { CameraLink } from './cameraLink'

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

type Props = {
  points: [number, number, number][]
  colors: string[]
  markers?: Marker[]
  highlight?: [number, number, number][][]
  onHover?: (index: number | null, x: number, y: number) => void
  onPick?: (index: number) => void
  link?: CameraLink
  linkId?: string
}

/** The chain as a coloured line in a WebGL canvas with orbit controls, markers and highlights. */
export const Scene = forwardRef<SceneHandle, Props>(function Scene({ points, colors, markers = [], highlight = [], onHover, onPick, link, linkId = 'scene' }, ref) {
  const controls = useRef<OrbitControlsImpl>(null)
  const gl = useRef<THREE.WebGLRenderer | null>(null)
  const { center, radius } = useMemo(() => {
    const box = new THREE.Box3().setFromPoints(points.map((p) => new THREE.Vector3(...p)))
    const sphere = box.getBoundingSphere(new THREE.Sphere())
    return { center: sphere.center, radius: Math.max(sphere.radius, 1) }
  }, [points])
  const shifted = useMemo(() => points.map(([x, y, z]) => [x - center.x, y - center.y, z - center.z] as [number, number, number]), [points, center])
  const vertexColors = useMemo(() => colors.map((c) => new THREE.Color(c).toArray() as [number, number, number]), [colors])
  useImperativeHandle(ref, () => ({
    reset: () => controls.current?.reset(),
    png: () => gl.current?.domElement.toDataURL('image/png') ?? null,
  }))
  const index = (e: ThreeEvent<PointerEvent | MouseEvent>) => (e.faceIndex ?? e.index ?? 0) as number
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
      <color attach="background" args={['#FFFFFF']} />
      <Line
        points={shifted}
        vertexColors={vertexColors}
        lineWidth={4}
        onPointerMove={(e) => {
          e.stopPropagation()
          onHover?.(index(e), e.nativeEvent.offsetX, e.nativeEvent.offsetY)
        }}
        onPointerOut={() => onHover?.(null, 0, 0)}
        onClick={(e) => {
          e.stopPropagation()
          onPick?.(index(e))
        }}
      />
      {highlight.map((seg, i) => (
        <Line key={i} points={seg.map(shift)} color="#151514" lineWidth={7} transparent opacity={0.25} />
      ))}
      {markers.length === 2 && <Line points={[shift(markers[0].at), shift(markers[1].at)]} color="#151514" lineWidth={1.5} />}
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
