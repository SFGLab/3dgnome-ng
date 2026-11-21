/** One question of a page to the server: whether it failed and with which message. */
type Asked = { isError: boolean; error: Error | null }

/**
 * The place of an answer that is not there. A question the server refused or did not answer shows
 * its message. A question still asked shows a busy mark.
 * @param of the questions the view waits for
 */
export function Unanswered({ of, className }: { of: Asked | Asked[]; className?: string }) {
  const failed = [of].flat().find((q) => q.isError)
  if (!failed) return <div className={className} aria-busy="true" />
  return (
    <div className={className}>
      <p className="field-error" role="alert">
        {failed.error?.message}
      </p>
    </div>
  )
}
