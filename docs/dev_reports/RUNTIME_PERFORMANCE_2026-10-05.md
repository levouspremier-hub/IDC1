# 主机修复后性能对照（初步）

固定同主机、同依赖、0.25秒预算、相同4次环境构造和8次历史失败快照。baseline固定8193d96，optimized固定307ebe4；均单槽且receipt验签通过。

|构造|基线秒|优化秒|
|---|---:|---:|
|冷5040|5.472739|4.247321|
|命中5040|0.039208|0.039584|
|新5088|5.379367|4.017134|
|新5136|5.429487|4.126658|

三次未缓存构造均值5.4272→4.1304秒，下降23.9%；含cProfile开销，不能作为最终训练速度。完整factory内部两次chain验证、每次字节fingerprint、缓存失效与深拷贝隔离保留。消除的仅为factory外重复chain调用。三日期provenance和8次快照执行结果将与回传记录逐项核对。

本地真实cached/uncached正式注入的价格/温度/碳/PV/风/arrival/task_specs/整数账本/refs/forecast及provenance逐项相同；独立环境reset观察逐位相同，任务对象互不共享。28项相关测试和Ruff/mypy通过。

基线8批尝试在第6批A超时停止，保留5成功批，累计Adam80、乘子5，失败批更新0，144条partial transitions；不补采成功结果。优化版8批独立尝试尚在主机运行。后续完整同协议吞吐、质量和512批估计见最终放行报告；当前不宣称长期资格或效率问题完全解决。

证据：runs/remote_runtime-repair-profile-baseline-v2/；runs/remote_runtime-repair-profile-optimized-v3/；runs/remote_runtime-repair-short-baseline-v2/。第一次候选清单失败 runs/remote_runtime-repair-profile-baseline-v1/ 保留。
