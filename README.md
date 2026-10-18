# maybenull.com

Source for [maybenull.com](https://maybenull.com), built with [Astro](https://astro.build) and deployed to GitHub Pages by GitHub Actions on every push to `main`.

## Writing a post

1. Copy `src/content/posts/hello-world.md` to a new file. The file name becomes the URL: `my-post.md` → `/posts/my-post/`.
2. Edit the title, date and text.
3. Set `draft: false` and push. The site redeploys in a minute or two.

## Running it locally

```sh
npm install
npm run dev      # live preview at http://localhost:4321
npm run build    # production build into dist/
```

## Layout

- `src/pages/index.astro`: the home page.
- `src/pages/posts/[...slug].astro`: the page template for each post.
- `src/layouts/Base.astro`: shared HTML head, styles and footer.
- `src/content/posts/`: posts in Markdown.
- `public/`: copied as-is. `CNAME` keeps the custom domain.
- `.github/workflows/deploy.yml`: builds and deploys the site.

DNS is at Namecheap: four A and four AAAA records on `@` point to GitHub Pages, and `www` is a CNAME to `griffonn.github.io`.
