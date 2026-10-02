# QM9 500 结构计算执行指南

版本 0.1  |  2026 年 9 月 21 日  |  用途：交给后续执行计算的人员或 agent 按步骤实施

目标是在已经选好的 500 个 QM9 几何上，建立可复现的 CCSD(T) 单点能量数据集。顺序为：核对输入与执行环境 → 固定并实现计算协议 → 50 个结构试跑 → 剩余 450 个结构跑批 → 验收与归档。50 个试跑结构属于这 500 个结构，不增加总规模。

当前已有结构和筛选元数据，尚未执行新的量子化学计算。本文给出后续执行规范；计算参数是建议的首版协议，完成第 3 节的接口核查后才能用于正式标注。

## 1  输入位置与启动前检查

存储更新（2026-10-01）：输入已校验归档至 Amarel 的 /home/your_username/datasets/qm9_ccsdt_seeds_500/（your_username 为归档所有者账号的占位符），本地 datasets/ 已删除。以下输入文件均相对于该归档目录；恢复到其他位置时保留内容和校验值。计算结果写入新的输出目录。归档与恢复说明见 docs/guides/repository-storage.md；本文计算状态仍为 9 月 21 日的历史记录。

|输入|用途|
|---|---|
|seed_geometries.json|500 个几何；读取顶层 seed_geometries 数组|
|xyz/ 与 seeds.xyz|单结构 XYZ 与合并多帧 XYZ；用于人工核对或转换|
|selection.json 与 manifest.json|分子 ID、尺寸、官能团和筛选来源|
|SHA256SUMS|迁移至执行主机后检查输入完整性|
|本指南目录中的 pilot50_ids.json|试跑子集，含原 40 个分子及新增的 10 个分子 ID|

- 确认恰好 500 个 molecule_id，无重复；所有输入 charge=0、spin=0、coordinate_unit=Angstrom。保留原子顺序、全精度坐标和原始 geometry_id。

- 几何来自 QM9 的 B3LYP/6-31G(2df,p) 优化。第一阶段只做固定几何单点，不重新优化，也不增加构型。

- 在执行主机保存 agent 提交号、PySCF/Python/NumPy 版本、BLAS、CPU、线程数、内存和 scratch 位置。每个 Run 使用独立 scratch 目录。

- 调用 get_capabilities 核实执行器。本次连接返回 local-process；若计划用集群，先完成远程执行器和资源配置，再重新验证，不能默认任务已送到集群。

启动条件：输入校验通过，输出根目录确定，资源上限写入记录，第 3 节的接口缺项已处理。此前只准备或验证任务，不提交 500 点跑批。

## 2  固定首版计算协议

建议协议标识为 qm9-ccsdt-def2svp-ae-v1。先采用全电子、常规积分的 RHF 参考态 CCSD(T)，减少方法混用。若改变基组、冻结芯层、积分近似或参考态，应创建新协议版本；所有样本按相同协议标注。

|项目|建议值或操作|
|---|---|
|任务与方法|single_point；method=ccsd_t；restricted=true|
|基组与体系|球谐 def2-SVP；charge=0；spin=0；symmetry=false|
|芯层与积分|所有电子参与相关；frozen=0；density_fitting.enabled=false|
|SCF 收敛|conv_tol=1e-10 Ha；conv_tol_grad=1e-6；max_cycle=100|
|CCSD 收敛|conv_tol=1e-8 Ha；conv_tol_normt=1e-7；max_cycle=100|
|参考态检查|显式请求内部和外部稳定性检查；分别保存结果|
|三重激发修正|只在 SCF 与 CCSD 均收敛后执行并接受 (T)|
|主要标签|HF 总能量、CCSD 总能量、(T) 修正、CCSD(T) 总能量|
|数值记录|Hartree；保存 float64 结果，展示时四舍五入不改变原始值|

这里的总能量指固定核坐标下的 Born–Oppenheimer 总能量，包含核核排斥，不含零点能或热修正。应满足 E_CCSD(T) = E_CCSD + E_(T)。QM9 的热力学属性不作为这批新标签的替代。

表中 SCF 参数已有对应运行字段；CCSD 专用阈值、迭代上限与显式 frozen 记录仍需完成绑定和验证，不能把它们写入任意 JSON 字段便声称已生效。收敛阈值控制数值求解，不代表有限基组结果达到 CBS 或化学精度。[1]

## 与 QH9 的关系

沿用几何、原子序数、坐标单位和来源等记录方式，新增 CCSD(T) 能量标签。QH9 的 B3LYP/def2-SVP Kohn–Sham 矩阵不能直接改名为 CCSD(T) 矩阵。若研究还需要 Fock/overlap，应另建同几何 DFT 任务，并记录 AO 顺序和相位约定。[2]

## 协议冻结时写下的资源设置

记录 execution_target、threads_per_task、memory_per_task、walltime、scratch、max_concurrency。试跑起步并发设为 1，测量内存和耗时后再提高。方法设置保持统一，资源可按尺寸分档；硬件未定时不填写虚构的总核时或完成日期。

## 3  先补齐接口并完成预检

本地核查依据为提交 74e805c。现有单点路径能够调用 CCSD 和 (T)，但以下要求仍须逐项实现或在实际执行环境中确认。本文没有修改计算后端。

- 参数生效：把 CCSD 的能量阈值、振幅阈值、max_cycle 和 frozen 显式传入求解器，导出 requested 与 effective 参数。runtime.conv_tol 当前设置的是 SCF，不是 CCSD。

- 阶段门控：SCF 未收敛时不启动正式 CCSD；CCSD 未收敛时不运行或接纳 (T)。现有调用段在事后汇总收敛状态，需补齐计算阶段之间的门控。

- 稳定性门控：将参考态检查安排在后 HF 计算之前，并保留两个方向的检查状态。当前诊断会在计算后汇总；仅设置 stable=true 不足以证明内部与外部检查都完成。

- 数据导出：保存独立的 CCSD 总能量、阶段耗时、实际参数、诊断和失败原因。若 CCSD 能量暂由总能量减去 (T) 得到，必须标注 derived，不能把该恒等式当独立数值验证。

## 当前可以做的接口验证

本目录 preflight_task_example.json 使用 qm9_000001 的原始坐标，已通过当前 MCP 的 validate_task。它仅验证现有接口，不包含尚未绑定的 CCSD 专用控制项，因此不是可直接接受为正式标签的生产输入。preflight_validation.json 保存本次校验摘要。

```text
validate_task(task_spec=<preflight_task_example.json 内容>, locale="zh")
```

启用稳定性检查的现有配置如下。analysis.outputs 只申请 energy；correlation_diagnostics 和 scf_stability 由模块生成，不是当前可直接填入 outputs 的请求项。

```text
"workflow": {"module_config": {
  "molecular.correlation_diagnostics": {"scf_stability": true}
}}
```

## 预检通过的判断

用至少 2 个小体系验证参数实际生效，检查阶段门控能拦截未收敛结果，并检查导出字段。数值检查按第 2 节执行；错误路径检查可用受控的迭代上限，不将失败结果写入正式标签表。验证后的任务模板和后端版本一并固定，再进入 50 点试跑。

validate_task 通过只证明输入与编排可解析，不能证明数值收敛、资源够用或方法适用。后续科学计算统一走 PySCF Agent 的执行入口。[3]

## 4  按批次执行

## 第一批为 50 个结构

读取 pilot50_ids.json，用 molecule_id 与 500 结构输入做连接，必须恰好得到 50 个唯一几何。清单保留原 40 个分子，补入 5 个最大 AO 样本和 5 个指纹差异较大的样本；覆盖完整集合 24–226 个 AO 的成本范围。

额外 10 个 QM9 数字 ID 为 121835、59979、6414、78226、39621、16548、3111、3655、16674、18039。实际读取时使用 JSON 中补零后的 molecule_id，不手工重建坐标。

- 先执行 2 个小体系做端到端检查，再执行成本尾部样本；资源配置确认后完成其余试跑点。预检结果仅在协议和全部质量要求一致时复用。

- 记录 SCF、稳定性、CCSD、(T) 和导出的耗时，以及队列等待、线程数、峰值内存和 scratch 峰值。按 AO/轨道数分档估算剩余计算，不仅用平均分子耗时。

- 选择至少 5 个试跑点作同协议独立复算，包含小体系、杂原子体系和高 AO 体系。参考任务使用独立输出目录；比对能量、实际设置和诊断。

## 使用现有 Study 工作流

把每个结构作为一个显式 case，使用 static Study，统一基任务模板；不要让自动方法路由将困难点的最终标签改为其他方法。先 validate_task，再 prepare_study 保存准备结果并记录 Study ID。核对 case 数、坐标、协议、执行器及资源估计后，对同一 Study 调用 submit_study。

用 get_study_status 跟踪；can_collect=true 后调用 collect_study，再 analyze_study 汇总诊断。查看已有单个 Run 时，可将原样保存的 JobHandle 交给 show_task 或 open_task_monitor；状态显示成功仍须通过第 5 节验收。[3]

## 通过试跑后补算剩余 450 个

若协议不变，复用试跑中已验收的记录，剩余 450 个建议按 25–50 个 case 分批准备。每批的并发和资源上限根据试跑结果设定；遇到同类系统性错误时停止启动后续批次，修复后恢复。

成功记录保持稳定 Task 标识；重试创建新 Run 并保留先前尝试。传输中断或提交结果不确定时，先检查原 Study、JobHandle 和执行器证据，不重新 prepare 同一批或盲目重复 submit_task。最终 500 个计划 ID 均须有可解释的终态。

进入跑批的条件：协议固定、参数生效有证据、预检和独立复算通过、失败路径可记录、最大体系资源可承受。出现协议变更时重新核验旧标签的可复用性。

## 5  验收与失败处理

每个结果归入 accepted、needs_review 或 rejected。调度器退出成功不等于科学验收通过；缺字段或诊断 unavailable 不当作通过。

|检查|接纳条件|
|---|---|
|身份与几何|molecule_id、geometry_id、坐标哈希与输入相符，协议一致|
|收敛|reference_converged=true 且 solver_converged=true|
|参考态|internal_stable=true 且 external_stable=true；缺失时待复核|
|能量|所有必需数值有限；(T) 存在且来源明确；单位与总能量定义一致|
|相关性诊断|T1/D1 和风险提示已保存；出现风险或证据不足时待复核|
|溯源|实际参数、程序版本、Run、日志及资源记录可追溯|

T1/D1 是风险指示，不能证明或否定能量精度。沿用当前 agent 的诊断规则时，保存规则版本和全部原始指标；风险提示触发 needs_review，避免仅凭单个阈值自动接受。SCF 或 CCSD 失败也不直接等同于强相关。

|失败类型|下一步|
|---|---|
|SCF 未收敛|保持方法与最终阈值；可调整初猜、DIIS 或 Newton 路径，并保存动作|
|CCSD 未收敛|检查参考态；在接口支持后调整迭代预算或 DIIS，不放宽验收阈值|
|内存不足或超时|按证据增加资源或减少并发；每次重试使用独立 Run 与 scratch|
|参考态不稳定或相关性风险|转入复核；其他参考态或多参考结果使用独立协议，不混入本主集|
|缺诊断或输出解析失败|从保留日志和工件补取；无法补齐则待复核，不填零或伪造成功|

建议每个 case 最多自动重试 2 次；仍失败则转入复核并记录最终原因。重试次数是首版运行政策，不是求解器内置默认值。不得用 MP2、CCSD 或较小基组结果填补缺失的 CCSD(T) 标签。

## 独立数值复核

完整 500 点阶段至少复核 20 个点，可包含先前的 5 个。覆盖不同尺寸和化学类别，并包含恢复成功的病例。使用同方法、基组、芯层、坐标和收敛设置；建议总能量差不超过 1e-6 Ha。该阈值用于复现检查，不表示真实物理误差。超限先排查参数、参考态、版本和导出。

## 6  导出结构与完成条件

建议在新的输出根目录保存 protocol.json、case_index.jsonl、labels.jsonl、attempts.jsonl、validation_summary.json 和原始运行工件。文件名及以下字段是本项目的导出约定，尚不是 agent 自动提供的现成数据格式。

|记录类别|最少字段|
|---|---|
|几何身份|molecule_id、geometry_id、geometry_hash、atomic_numbers、positions、coordinate_unit|
|方法与能量|protocol_id、basis、reference、frozen、effective_parameters、E_HF、E_CCSD、E_T、E_CCSD_T、energy_unit|
|科学质量|SCF/CCSD 收敛、内部/外部稳定性、T1/D1、warnings、acceptance_status、reason|
|运行与成本|study_id、case_id、task_id、run_id、JobHandle、资源、分阶段耗时、版本、日志路径|
|恢复证据|attempt_index、failure_type、recovery_action、parent_run_id、人工介入次数|

当前结果中的 reference_energy 对应 HF 总能量；triples_correction 对应 E_T。对 ccsd_t 任务，correlation_energy 包含 (T)，不能直接作为纯 CCSD 相关能使用。标签映射必须核对原始结果和求解器日志。[3]

- case_index 覆盖全部 500 个计划 ID。labels 只包含通过验收的记录；每个 molecule_id、geometry_id、protocol_id 组合只计一个最终标签，所有尝试保留在 attempts。

- 汇总计划数、accepted、needs_review、rejected，三类之和必须等于 500。若 accepted 少于 500，应如实报告缺口；不能把重试次数计成结构数。

- 保存 20 点复核结果、失败原因分布、耗时和资源统计，以及全套文件校验值。若后续按分子分训练/验证/测试集，400/50/50 可作为预先固定的方案；同一分子的所有后续构型放在同一集合。

## 交给下一位执行者的首个任务

先完成第 3 节的参数绑定、阶段门控与导出检查，再将 pilot50_ids.json 转成 50 个静态 Study case，逐个验证，保存可审阅的任务清单与资源计划。随后按第 4 节试跑和扩展；当前文档及输入校验没有启动任何科学计算。

多构型生成属于后续独立阶段。待这 500 点流程稳定后，再确定采样方法和预算；当前 prepare_dataset 对应 B3LYP 的 MD 路径，不能把方法字符串替换为 CCSD(T) 就当作已支持。

## 依据与相关文件

[[1] PySCF Coupled cluster theory](https://pyscf.org/user/cc.html)

[[2] QH9 数据集论文](https://arxiv.org/html/2306.09549v4)

[3] 本地接口及实现：docs/guides/mcp.md；pyscf_agent/contracts.py；pyscf_agent/backend/execution.py；pyscf_agent/backend/correlation/molecular.py。以上结论对应 74e805c；执行前复查实际部署版本。
