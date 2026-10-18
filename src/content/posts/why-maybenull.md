---
title: Why maybenull
description: Object reference not set to an instance of an explanation.
date: 2026-10-19
draft: false
---

```csharp
#nullable enable

string? reason = GetReasonForThisWebsite();
Console.WriteLine(reason.Length);
```

The compiler tried to warn me:

```
warning CS8602: Dereference of a possibly null reference.
```

I shipped it anyway.

```
Unhandled exception. System.NullReferenceException:
Object reference not set to an instance of an object.
```

So, why maybenull? `reason` was null. It's always been null.

Fix planned for the next release. Probably.
