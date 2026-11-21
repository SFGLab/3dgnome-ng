/** Saves a file from the page to the visitor's computer. */

/** Saves what a URL of the page holds, a data address or an object address, under a file name. */
export function saveUrl(url: string, name: string): void {
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.click()
}

/** Saves a blob under a file name. */
export function saveBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob)
  saveUrl(url, name)
  URL.revokeObjectURL(url)
}
