import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'

/** A folder of src that must not import the folders named: the rule and its message. */
const apart = (folders, message) => ({
  'no-restricted-imports': ['error', { patterns: [{ group: folders.flatMap((f) => [`**/${f}`, `**/${f}/**`]), message }] }],
})
const SHARED = 'Code that two parts of the site use is in api/, lib/, components/ or content/.'

export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
    },
  },
  // The folders of src import in one direction. A page puts the parts of the site together, and a
  // part does not reach into another part or into a page.
  { files: ['src/job/**'], rules: apart(['model', 'pages'], `job/ does not import the form or a page. ${SHARED}`) },
  { files: ['src/model/**'], rules: apart(['job', 'pages'], `model/ does not import the job views or a page. ${SHARED}`) },
  { files: ['src/myjobs/**'], rules: apart(['job', 'model', 'pages'], `myjobs/ does not import the form, the job views or a page. ${SHARED}`) },
  {
    files: ['src/api/**', 'src/lib/**', 'src/components/**', 'src/content/**'],
    rules: apart(['job', 'model', 'myjobs', 'pages'], 'The shared folders do not import a part of the site or a page.'),
  },
)
