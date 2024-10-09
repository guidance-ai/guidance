import svelte from 'rollup-plugin-svelte';
import { sveltePreprocess } from 'svelte-preprocess';
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import terser from '@rollup/plugin-terser';
import typescript from '@rollup/plugin-typescript';
import postcss from 'rollup-plugin-postcss';
import livereload from 'rollup-plugin-livereload';
// @ts-ignore
import serve from 'rollup-plugin-serve';
// @ts-ignore
import htmlBundle from 'rollup-plugin-html-bundle';
import copy from 'rollup-plugin-copy';

const production = !process.env.ROLLUP_WATCH;

export default [{
	input: 'src/main.js',
	output: {
		format: 'iife',
		name: 'app',
		file: 'build/bundle.js',
		sourcemap: !production,
	},
	plugins: [
		typescript(),
		svelte({
			compilerOptions: {
				dev: !production
			},
			preprocess: sveltePreprocess()
		}),
		resolve({
			browser: true,
			dedupe: importee => importee === 'svelte' || importee.startsWith('svelte/'),
			extensions: ['.svelte', '.mjs', '.ts', '.js', '.json', '.node']
		}),
		commonjs(),
		postcss(),
		copy({
			targets: [
				{ src: 'src/template.html', dest: 'build' }
			]
		}),
		htmlBundle({
			template: 'build/template.html',
			target: production ? 'dist/index.html' : 'build/index.html',
			targetElement: 'body',
			inline: production
		}),
		!production && serve('build'),
		!production && livereload('build'),
		production && terser()
	],
	watch: {
		clearScreen: false
	}
}];