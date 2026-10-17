# Python asyncio 异步编程

asyncio 是 Python 的单线程异步框架，用 async/await 写并发。

## 核心概念

- `async def` 定义协程
- `await` 挂起当前协程，让出控制权
- `asyncio.run()` 启动入口
- `asyncio.gather()` 并发跑多个协程

## 适用场景

适合 IO 密集（网络请求、文件读写），不适合 CPU 密集。用事件循环在单线程里切换任务，避免多线程的开销。
