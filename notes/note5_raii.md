# C++ RAII 与智能指针

RAII（Resource Acquisition Is Initialization）是 C++ 管理资源的惯用法：把资源的获取和释放绑定到对象的生命周期。

## 核心思想

- 构造函数里获取资源，析构函数里释放资源
- 对象离开作用域自动析构，资源自动释放，异常安全

## 智能指针

- `unique_ptr`：独占所有权，不能拷贝
- `shared_ptr`：共享所有权，引用计数
- `weak_ptr`：不增加引用计数，用来打破循环引用
