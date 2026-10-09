# MMV2010 对照：当前新 μ DeepONet 口径

修改日期：2026-09-29；2026-09-30复核。本实现参考 Maliar, Maliar and Valli (2010), “Solving the incomplete markets model with aggregate uncertainty using the Krusell–Smith algorithm”, Journal of Economic Dynamics and Control 34, 42–49, DOI: 10.1016/j.jedc.2009.03.009，以及作者公开的 MAIN.M、INDIVIDUAL.m、SHOCKS.m、AGGREGATE_ST.m。作者源代码被用于核对算法；这是新的 Python/SciPy 适配实现，不是原 MATLAB 程序逐位复现。

原始软件完整 ZIP、许可证和来源哈希在 `docs/mmv2010_author_source/`。保留其中原作者的历史测试输入仅为完整归档；程序不加载这些输入，不冒充当前同口径对照。遵循原许可，用于非商业研究和教学；论文需突出说明使用来源并引用原文章。

## 实验要回答什么

MMV 用四维政策表 s(a,e,K,z)，再拟合两条 log K′ = b0(z)+b1(z) log K。DeepONet 读取完整离散联合 μ 与 K。两者解决同一 Model B 校准，比较当前共同状态精度、同 K 形状差异、各自动态表现与实际求解成本。传统方法在模拟中也保存财富分布；“能输入 μ”本身不能证明普遍创新或精度优势。共同函数库包含外生人工状态，不能称作均衡可达状态。

主实现是论文 Algorithm 1（随机个体模拟），不是 Algorithm 2（作者逆 CDF 非随机模拟）。当前共同评价使用已有区间运输，不把它称为原作者 Algorithm 2。

## 论文到代码的对应

1. 个体资产网格 a_i = A (i/(n−1))^7。给定 B，用阻尼 Euler 固定点求政策；原始更新最大误差≤1e−8，阻尼0.7。初始储蓄0.9a，并投影到当前可行预算。
2. 给定固定的10,000个体、1,100期冲击面板模拟资本，丢弃前100期，按当前生产率分别 OLS 回归 log K′。每次外循环重用同一随机流。
3. B 阻尼0.3更新，原始系数差 L2≤1e−8才标记收敛。原作者代码使用此范数，与论文文字中的平方误差描述有区别。外循环预算500、内循环预算20,000；预算耗尽保留失败，不出“已收敛”结果。
4. 当原始系数差>1e−6时，以本次模拟末财富面板更新下一次模拟初态，随后冻结该面板。初始面板从官方联合分布抽样，而不是作者源码的全体同资产初态；这一差异记录在配置中。回归中的当期K由当前个体资产均值计算，不以感知ALM代替。
5. 收敛后再次在最终 B 上求政策，保存政策表、B、每次迭代、冲击哈希、实际时间和进程内存。每次外循环原子保存 checkpoint；内循环中断后从最后完整外循环恢复，损失工作时间单独记录。

## 冻结的五个配置

| 配置 | 种子 | 资产上界/资产节点 | K节点 | 用途 |
|---|---|---|---|---|
| matched | 9711–9713 | 500 / 100 | 27–50均匀4点 | 新 DeepONet 主口径对照 |
| fine | 9711 | 500 / 300 | 27–50均匀4点 | 政策表资产网格敏感性 |
| author_domain | 9711 | 1000 / 100 | 30–50均匀4点 | 作者求解域敏感性，单列结果 |

新函数库原始目标 K≈29.33–44.00，局部扰动再±2%；因此不能照搬作者30的下界后悄悄裁剪当前状态。主配置预先固定27–50，没有根据模型分数调整。实际查询的 K 超域直接失败并保留；作者代码中感知资本预测的边界裁剪予以保留且统计次数。感知预测裁剪不等于改变评价时的实际 K。

SciPy RectBivariateSpline 使用张量三次插值，不保证与 MATLAB interpn('cubic') 一致。负储蓄、超资产上界和消费预算投影均统计。消费下界采用当前 DeepONet 1e−4；不同于原作者某些运算的1e−10保护值。MMV求解采用官方人口边际计算价格；独立评价按当前 μ 的实际人口边际计算预算，并记录可行性投影。rounded转移概率产生的小边际差异不得解释为算法精度提升。

作者域配置使用1000的评价运输域；低于30或一步后超域的函数将失败，覆盖率完整保留。不能与500主配置平均数混用。政策表加密与 DeepONet 输入网格加密并非相同数值操作；本包的 fine 是传统政策表加密，后续仍需两种方法共同的运输网格复评。

## 评价与成本

默认每个配置收敛后做开发集静态评价：共享5400状态，其中1800原始父函数按6个生成组汇总，局部扰动单列；同 K 检验报告两端 KKT 和储蓄响应。概率权重、零原子和价格均沿用未改动的共同核心。

主误差由实际政策推进 μ′ 后计算 Euler/KKT。MMV 感知 ALM 下的 KKT、ALM预测与实际推进 K′ 的差异仅为附加诊断。恒等构造的质量、预算等指标标为数值检查；P99 是误差分位数，不是高财富组误差。

动态评价另行启动，读取相同6条开发路径，每条4000期，从官方初态独立推进。前1000与后3000分别汇总；失败轨迹、日期、覆盖率保留，不能把成功子集平均当作完整主结果。最终测试没有CLI入口；本包不打开3600测试状态或120对最终形状状态，也不执行最终压力测试。投稿前仍须冻结跨方法最终 checkpoint 名单和新的比较协议。

MMV 主要在 CPU float64 上运行，共同评价在同机 GPU float32 上运行。用同一张2080Ti主机满足硬件环境统一，但不把传统 CPU 算法包装为 GPU 算法。传统迭代按收敛条件停止，不强套100k梯度更新或神经网络参数量。保存软件、GPU/CPU、线程数、阶段时间、内存和中断记录；时间口径分求解、开发评价、数据准备，DeepONet的100k训练时间另取原日志。速度、精度优势均等待实测。

## AutoDL 可复制命令

先上传 `server/mmv2010_jedc_v1_upload_20260930_r2.zip` 到 `/root/rivermind-data/`。以下创建独立目录，不修改正在运行的 DeepONet 文件。

```bash
cd /root/rivermind-data
python - <<'PY'
import zipfile, pathlib, json, hashlib
archive=pathlib.Path('mmv2010_jedc_v1_upload_20260930_r2.zip')
root=pathlib.Path('mmv2010_jedc_v1_review_20260930')
if root.exists(): raise RuntimeError('目录已存在，请进入该目录恢复，禁止覆盖')
with zipfile.ZipFile(archive) as z:
    for n in z.namelist():
        p=pathlib.PurePosixPath(n)
        assert not p.is_absolute() and '..' not in p.parts, n
    z.extractall(root)
m=json.loads((root/'UPLOAD_MANIFEST.json').read_text())
for n,h in m['files'].items():
    assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
print('MMV上传包所有文件哈希通过；尚未求解')
PY
cd /root/rivermind-data/mmv2010_jedc_v1_review_20260930
python -m pip install 'numpy>=1.24' 'scipy>=1.10'
REF=/root/rivermind-data/mu_functions_v2_main9_20260929/config/mu_functions_v2_mu971_main9_20260929_fix1/experiment_manifest.json
python -u server/run_mmv2010.py --reference-plan "$REF" --tag mmv971_matched_review_20260930 --prepare-only
```

准备入口核验已有新 μ 数据与共同源代码哈希，冻结配置；不训练、不评价。等 DeepONet 队列与其他实验空闲后继续：

```bash
ps -eo pid,etime,pcpu,args | grep -E 'train_jedc|run_jedc|train_mmw|run_mmw|solve_mmv2010' | grep -v grep || true
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/test_mmv2010.py
bash server/start_mmv2010.sh mmv971_matched_review_20260930 "$REF" matched
tail -f runs/mmv2010_queue_mmv971_matched_review_20260930/matched_s9711.log
```

日志尚未创建时先看启动脚本打印的总日志路径。只有出现 `MMV outer ... individual ...` 才确认个体求解开始，出现 `MMV OUTER` 才确认完成了一个聚合迭代。Ctrl+C仅停止tail。启动脚本检测既有binding自动要求恢复，求解器检查其他活跃实验；检测自身祖先进程时排除包装脚本，避免此前准备器误判自身。

fine与作者域不是默认三次主实验的一部分，单独执行：

```bash
bash server/start_mmv2010.sh mmv971_matched_review_20260930 "$REF" fine
bash server/start_mmv2010.sh mmv971_matched_review_20260930 "$REF" author_domain
```

两条命令应依次执行，前一队列结束后再启动后一队列。

主配置的自身动态评价（先等主队列结束，每个seed依次运行）：

```bash
for SEED in 9711 9712 9713; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u scripts/evaluate_mmv2010.py \
    --binding config/mmv2010_mmv971_matched_review_20260930/binding.json \
    --run-dir runs/mmv2010_queue_mmv971_matched_review_20260930/matched_s${SEED}
done
```

## 当前核验状态

2026-09-30复核版另核对原作者 SHOCKS.m 的联合转移、MAIN.M 的七次幂网格及收敛阈值，并修复完成运行的checkpoint/同卡核验。原2026-09-29上传包保留，不与复核版混用。Windows本机只执行AST语法、JSON、来源哈希、ZIP完整性和共享源码一致性检查。没有执行求解、数值单测、开发评价或生成科研图。云端运行是正式实验前仍需完成的数值核验；代码尚不能称作已验证重现了论文数值表。

三次新 DeepONet 与三次 matched 的开发审计均完整后，生成配对组指标和单列成本表：

```bash
python -u scripts/summarize_mmv2010.py \
  --binding config/mmv2010_mmv971_matched_review_20260930/binding.json \
  --queue runs/mmv2010_queue_mmv971_matched_review_20260930 \
  --deep-tag mu971_main9_20260929_fix1 \
  --output paper/mmv2010_development_comparison
```

汇总器要求每个模型5400个开发状态全部对应且无失败，核验评价数据与checkpoint；出现不完整覆盖会停止并保留原审计，不自动筛掉失败。输出18行（3个种子×6个生成组）的原始父函数配对结果，不把1800个相关函数当作独立显著性样本。同K响应和各自动态CSV另表呈现；汇总器不替代这些检验，也不读取最终测试。
