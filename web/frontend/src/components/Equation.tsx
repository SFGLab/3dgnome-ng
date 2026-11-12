import sizes from '../assets/eq/sizes.json'
import tex from '../assets/eq/tex.json'

const IMAGES = import.meta.glob<string>('../assets/eq/*.png', { eager: true, import: 'default' })

export type EquationName = keyof typeof tex

/** A typeset equation from web/design/render_equations.py at its display size, with its TeX as alt text. */
export function Equation({ name }: { name: EquationName }) {
  const [w, h] = sizes[name]
  return <img className="equation" src={IMAGES[`../assets/eq/${name}.png`]} width={w} height={h} alt={tex[name]} />
}
