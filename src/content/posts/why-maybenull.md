---
title: Why maybenull
description: On naming a website after a value that might not be there.
date: 2026-10-19
draft: false
---

Every programmer has met `null`: the value that isn't there. Most of us met it
the hard way, as a crash at 2am.

Modern languages got more honest about it. Kotlin and Swift write `String?`,
Java has `Optional<T>`, Haskell has `Maybe a`. These types admit up front that
the answer might be missing, and they make you deal with that case instead of
pretending it can't happen.

That felt like the right name for a personal site. Not because nothing will be
here (though for a while the posts list on the home page literally said
`null`), but because most of the things I find interesting start with not
knowing. A good question is a nullable value: you don't know yet whether
there's an answer, and the honest move is to say so.

The name itself came out of a long search for something like "no clue". Along
the way it went through priors, p-values, overfitting and "still converging",
and `maybenull` stuck: it's short, it's a .com, and it's a joke that compiles.

So what will live here? Notes on things I'm figuring out, half-finished ideas,
and now and then something that turns out to be right. Treat every post as
optional, and check before you dereference.

```kotlin
val post: Post? = maybenull.latest()
post?.read() ?: println("still converging")
```
