from __future__ import annotations

from pyscf_agent.resources.web import versioned_assets

import html
import json
import logging
from typing import Any, Sequence

from pyscf_agent.paths import resolve_work_dir
from pyscf_agent.backend.workflow import example_request
from pyscf_agent.registry import default_registry
from pyscf_agent.registry.platform import (
    DEFAULT_ANALYSIS,
    DEFAULT_PERIODIC_BASIS_SET,
    DENSITY_FITTING_AUXBASIS_RECOMMENDATIONS,
)
from pyscf_agent.web_assets.template import HTML_PAGE


LOGGER = logging.getLogger(__name__)
UI_OPTION_LABEL_KEYS = {
    'task_type.molecular': 'taskFamilyMolecular',
    'task_type.model_hamiltonian': 'taskFamilyModelHamiltonian',
    'task_type.periodic': 'taskFamilyPeriodic',
    'job.single_point': 'jobSinglePoint',
    'orbital_processing.none': 'localizationNone',
    'orbital_processing.boys': 'localizationBoys',
    'orbital_processing.pipek_mezey': 'localizationPipekMezey',
    'active_space.manual': 'activeSpaceManual',
    'active_space.occupation_window': 'activeSpaceOccupationWindow',
    'active_space.energy_window': 'activeSpaceEnergyWindow',
    'active_space.avas': 'activeSpaceAvas',
    'periodic_band_path.auto': 'periodicBandPathModeAuto',
    'periodic_band_path.seekpath': 'periodicBandPathModeSeekpath',
    'periodic_band_path.custom': 'periodicBandPathModeCustom',
    'periodic_band_path.explicit': 'periodicBandPathModeExplicit',
}
METHOD_LABELS = {
    'dft': 'DFT',
    'ccsd_t': 'CCSD(T)',
}
MODEL_SOLVER_LABELS = {
    'fci': 'Full CI',
    'ccsd_t': 'CCSD(T)',
}
ACTIVE_SPACE_SOLVER_LABELS = {
    'fci': 'Full CI',
    'block2_dmrg': 'block2 DMRG',
}
OUTPUT_LABELS = {
    'energy': '总能',
    'homo_lumo': 'HOMO/LUMO',
    'dipole': '偶极矩',
}
MODEL_OUTPUT_LABELS = {
    'strong_correlation_diagnostics': '强关联诊断',
}
PERIODIC_OUTPUT_LABELS = {
    'energy': '每晶胞总能',
    'band_gap': '平均场带隙',
    'fermi_energy': '费米能估计',
    'band_structure': '高对称路径能带',
}
BASIS_OPTIONS = tuple(
    item.id
    for item in default_registry().option_values(
        'options.molecular.basis', backend_allowed=True
    )
)
AUXBASIS_LABELS = {
    '__off__': 'No density fitting',
    '': 'PySCF auto',
    'weigend+etb': 'weigend+etb',
    'def2-universal-jkfit': 'def2-universal-jkfit',
    'def2-svp-jkfit': 'def2-svp-jkfit',
    'def2-tzvp-jkfit': 'def2-tzvp-jkfit',
    'def2-qzvp-jkfit': 'def2-qzvp-jkfit',
    'cc-pvdz-jkfit': 'cc-pvdz-jkfit',
    'cc-pvtz-jkfit': 'cc-pvtz-jkfit',
    'aug-cc-pvdz-jkfit': 'aug-cc-pvdz-jkfit',
    'heavy-aug-cc-pvdz-jkfit': 'heavy-aug-cc-pvdz-jkfit',
}
AUXBASIS_OPTIONS = tuple(
    [('__off__', AUXBASIS_LABELS['__off__']), ('', AUXBASIS_LABELS[''])] + [
        (
            capability.id,
            AUXBASIS_LABELS.get(capability.id, capability.label),
        )
        for capability in default_registry().option_values(
            'options.molecular.auxbasis', backend_allowed=True
        )
    ]
)
METHOD_OPTIONS = tuple(
    (
        capability.id,
        METHOD_LABELS.get(capability.id, capability.label),
    )
    for capability in default_registry().capabilities(
        namespace='molecular.method', backend_allowed=True
    )
)
XC_OPTIONS = tuple(
    item.id
    for item in default_registry().option_values(
        'options.molecular.xc', backend_allowed=True
    )
)
MODEL_SOLVER_OPTIONS = tuple(
    (
        capability.id,
        MODEL_SOLVER_LABELS.get(capability.id, capability.label),
    )
    for capability in default_registry().capabilities(
        namespace='model_hamiltonian.solver',
        backend_allowed=True,
        ui_visible=True,
    )
)
DMET_REFERENCE_DENSITY_OPTIONS = tuple(
    (capability.id, capability.label)
    for capability in default_registry().capabilities(
        namespace='embedding.reference_density',
        backend_allowed=True,
        ui_visible=True,
    )
)
PERIODIC_METHOD_OPTIONS = tuple(
    (capability.id, METHOD_LABELS.get(capability.id, capability.label))
    for capability in default_registry().capabilities(
        namespace='periodic.method', backend_allowed=True
    )
)
PERIODIC_CORRELATION_OPTIONS = (
    ('none', 'Mean-field only'),
) + tuple(
    (capability.id, capability.label)
    for capability in default_registry().capabilities(
        namespace='embedding.method', backend_allowed=True, ui_visible=True
    )
    if capability.id in ('gw', 'hf_dmft', 'gw_dmft')
)
PERIODIC_DMFT_IMPURITY_SOLVER_OPTIONS = tuple(
    (capability.id, capability.label)
    for capability in default_registry().capabilities(
        namespace='embedding.dmft_impurity_solver',
        backend_allowed=True,
        ui_visible=True,
    )
)
PERIODIC_DMFT_LOCALIZATION_OPTIONS = tuple(
    (capability.id, capability.label)
    for capability in default_registry().capabilities(
        namespace='embedding.localization', backend_allowed=True, ui_visible=True
    )
    if capability.id in ('iao', 'iao_pao')
)
PERIODIC_BASIS_OPTIONS = tuple(
    (capability.id, capability.label)
    for capability in default_registry().option_values(
        'options.periodic.basis', backend_allowed=True
    )
)
PERIODIC_PSEUDO_OPTIONS = tuple(
    (
        capability.id,
        '{0} ({1})'.format(capability.label, capability.constraints['element_range'])
        if capability.constraints.get('general') and capability.constraints.get('element_range')
        else capability.label,
    )
    for capability in default_registry().option_values(
        'options.periodic.pseudopotential', backend_allowed=True
    )
)
PERIODIC_XC_OPTIONS = tuple(
    (capability.id, capability.label)
    for capability in default_registry().option_values(
        'options.periodic.xc', backend_allowed=True
    )
)
PERIODIC_DENSITY_FITTING_OPTIONS = tuple(
    (capability.id, capability.label)
    for capability in default_registry().option_values(
        'options.periodic.density_fitting', backend_allowed=True
    )
)
PERIODIC_KPOINT_SCHEME_OPTIONS = tuple(
    (capability.id, capability.label)
    for capability in default_registry().option_values(
        'options.periodic.kpoint_scheme', backend_allowed=True
    )
)
PERIODIC_BAND_PATH_MODE_OPTIONS = tuple(
    (
        capability.id,
        capability.label,
        UI_OPTION_LABEL_KEYS.get('periodic_band_path.{0}'.format(capability.id)),
    )
    for capability in default_registry().option_values(
        'options.periodic.band_path_mode', backend_allowed=True
    )
)
PERIODIC_SMEARING_OPTIONS = tuple(
    (capability.id, capability.label)
    for capability in default_registry().option_values(
        'options.periodic.smearing', backend_allowed=True
    )
)
PERIODIC_EXXDIV_OPTIONS = tuple(
    (capability.id, capability.label)
    for capability in default_registry().option_values(
        'options.periodic.exxdiv', backend_allowed=True
    )
)
OUTPUT_OPTIONS = tuple(
    (
        capability.id,
        OUTPUT_LABELS.get(capability.id, capability.label),
    )
    for capability in default_registry().observables(
        namespace='molecular.observable',
        requestable=True,
        backend_allowed=True,
    )
    if capability.metadata.get('assistant_ui_output', True)
)
MODEL_OUTPUT_OPTIONS = tuple(
    (
        capability.id,
        MODEL_OUTPUT_LABELS.get(capability.id, capability.label),
    )
    for capability in default_registry().observables(
        namespace='model_hamiltonian.observable',
        requestable=True,
        backend_allowed=True,
    )
    if capability.metadata.get('assistant_ui_output', True)
)
PERIODIC_OUTPUT_OPTIONS = tuple(
    (
        capability.id,
        PERIODIC_OUTPUT_LABELS.get(capability.id, capability.label),
    )
    for capability in default_registry().observables(
        namespace='periodic.observable',
        requestable=True,
        backend_allowed=True,
    )
    if capability.metadata.get('assistant_ui_output', True)
)
TASK_FAMILY_OPTIONS = tuple(
    (
        capability.id,
        capability.label,
        UI_OPTION_LABEL_KEYS.get('task_type.{0}'.format(capability.id)),
    )
    for capability in default_registry().capabilities(
        namespace='task_type', backend_allowed=True
    )
)
JOB_OPTIONS = tuple(
    (
        capability.id,
        capability.label,
        UI_OPTION_LABEL_KEYS.get('job.{0}'.format(capability.id)),
    )
    for capability in default_registry().capabilities(
        namespace='molecular.job', backend_allowed=True
    )
)
LOCALIZATION_METHOD_OPTIONS = tuple(
    [('none', 'No localization', UI_OPTION_LABEL_KEYS['orbital_processing.none'])] + [
        (
            capability.id,
            capability.label,
            UI_OPTION_LABEL_KEYS.get('orbital_processing.{0}'.format(capability.id)),
        )
        for capability in default_registry().capabilities(
            namespace='molecular.orbital_processing', backend_allowed=True
        )
        if capability.metadata.get('localization_method') is True
    ]
)
ACTIVE_SPACE_METHOD_OPTIONS = tuple(
    (
        capability.id,
        capability.label,
        UI_OPTION_LABEL_KEYS.get('active_space.{0}'.format(capability.id)),
    )
    for capability in default_registry().capabilities(
        namespace='molecular.active_space', backend_allowed=True
    )
)
ACTIVE_SPACE_SOLVER_OPTIONS = tuple(
    (
        capability.id,
        ACTIVE_SPACE_SOLVER_LABELS.get(capability.id, capability.label),
    )
    for capability in default_registry().capabilities(
        namespace='molecular.active_space_solver', backend_allowed=True
    )
)
UI_TRANSLATIONS = {
  'zh': {
    'htmlLang': 'zh-CN',
    'pageTitle': 'PySCF 计算助手',
    'toggleLabel': 'English',
    'heroTitle': 'PySCF 计算助手',
    'heroDescription': '将自然语言整理为可审查的 PySCF 请求，并通过预定义 workflow 执行计算。',
    'openStudyAgent': 'Planner Agent',
    'configTitle': '任务配置',
    'configDescription': '',
    'labelTaskFamily': '体系',
    'taskFamilyMolecular': '分子电子结构',
    'taskFamilyModelHamiltonian': '模型哈密顿量',
    'taskFamilyPeriodic': '周期电子结构',
    'taskSessionsTitle': '任务',
    'newTask': '新建任务',
    'taskSessionMolecular': '分子任务',
    'taskSessionModelHamiltonian': '模型任务',
    'taskSessionPeriodic': '周期任务',
    'taskStatusDraft': '草稿',
    'taskStatusReady': '待计算',
    'taskStatusChanged': '配置已更改',
    'taskStatusRunning': '计算中',
    'taskStatusCancelled': '已中止',
    'taskStatusCompleted': '已完成',
    'taskStatusFailed': '失败',
    'taskStarted': '已开始新的任务会话',
    'builderInputLinked': 'Builder 生成的模型哈密顿量输入已关联到当前任务。',
    'workDirLockedHint': '已锁定到实际运行目录',
    'labelWorkDir': '工作目录',
    'labelExecutionTarget': '执行位置',
    'labelResourceProfile': '资源配置',
    'workDirPlaceholder': '留空则使用当前目录下的 runs/',
    'effectiveWorkDirLabel': '正在使用',
    'labelModelSolver': '模型求解器',
    'previewModelHamiltonian': '预览结构',
    'modelHamiltonianPreviewTitle': '模型结构预览',
    'modelHamiltonianPreviewEmpty': '用 builder 生成输入文件后，可以在这里预览模型结构图。',
    'modelHamiltonianPreviewLoading': '正在读取模型结构...',
    'modelHamiltonianPreviewFailed': '模型结构预览失败',
    'modelHamiltonianPreviewTruncated': '图形预览已截断，仅显示前若干个位点和键。',
    'modelHamiltonianPreviewNoSites': '输入文件中没有可绘制的位点。',
    'openModelHamiltonianBuilder': '打开模型哈密顿量 Builder',
    'modelHamiltonianMissingFile': '请先用 builder UI 生成模型哈密顿量 PySCF 输入文件。',
    'modelHamiltonianPrepared': '模型哈密顿量请求已整理完成，可以确认计算。',
    'periodicStructureTitle': '周期结构',
    'labelPeriodicFormat': '结构格式',
    'labelPeriodicFile': '读取结构文件',
    'labelPeriodicStructure': 'POSCAR / CIF 内容',
    'periodicStructurePlaceholder': '上传或粘贴 POSCAR/CIF 内容',
    'previewPeriodicStructure': '检查结构',
    'periodicPreviewEmpty': '上传或粘贴 POSCAR/CIF 后检查结构。',
    'periodicPreviewLoading': '正在解析周期结构...',
    'periodicPreviewFailed': '周期结构解析失败',
    'labelPeriodicBasis': '周期基组',
    'labelPeriodicPseudo': '赝势',
    'labelPeriodicMethod': '计算方法',
    'labelPeriodicXc': '交换关联泛函',
    'labelPeriodicKmesh': 'k 点网格',
    'labelPeriodicKpointScheme': 'k 点方案',
    'labelPeriodicBandPathMode': '能带路径',
    'periodicBandPathModeAuto': '自动',
    'periodicBandPathModeSeekpath': 'SeeK-path 标准原胞',
    'periodicBandPathModeCustom': '自定义标准点',
    'periodicBandPathModeExplicit': '显式坐标',
    'labelPeriodicBandPath': '高对称点路径',
    'periodicBandPathPlaceholder': '例如 GXWKGLUWLK,UX',
    'labelPeriodicBandSpecialPoints': '高对称点约化坐标',
    'periodicBandSpecialPointsPlaceholder': 'G = 0, 0, 0\nX = 0.5, 0, 0.5',
    'labelPeriodicBandPathNpoints': '能带路径采样点',
    'labelPeriodicBandPathReferenceDistance': '路径点间距（1/Angstrom）',
    'labelPeriodicBandPathSymprec': '对称性容差（Angstrom）',
    'periodicNumericsTitle': '数值控制',
    'labelPeriodicKpointShift': 'k 点位移（网格步长）',
    'labelPeriodicPrecision': '积分精度',
    'labelPeriodicKeCutoff': '平面波辅助截断（Hartree）',
    'labelPeriodicFftMesh': 'FFT 网格（留空则自动）',
    'labelPeriodicDensityFitting': '密度拟合',
    'labelPeriodicDfAuxbasis': '辅助高斯基组（可选）',
    'periodicDfAuxbasisPlaceholder': '仅用于 GDF / MDF；留空则自动',
    'labelPeriodicExxdiv': '精确交换发散处理',
    'labelPeriodicSmearing': '占据展宽',
    'labelPeriodicSmearingSigma': '展宽 sigma（Hartree）',
    'labelPeriodicSmearingFixSpin': '分别固定 alpha / beta 电子数',
    'labelPeriodicSpin': '自旋 (Nalpha - Nbeta)',
    'periodicMissingStructure': '请先上传或粘贴 POSCAR/CIF 结构。',
    'periodicPrepared': '周期计算请求已整理完成，可以确认计算。',
    'labelAtom': '分子结构',
    'sectionStructure': '结构',
    'sectionCalculationSettings': '计算设置',
    'sectionSpinParameters': '电荷与自旋',
    'sectionActiveSpace': 'Active Space',
    'labelBasis': '基组',
    'labelMethod': '计算方法',
    'labelXc': '交换关联泛函',
    'labelJob': '任务类型',
    'jobSinglePoint': 'single point',
    'advancedParameters': '高级参数',
    'labelCharge': '总电荷',
    'labelSpin': '自旋 (Nα - Nβ)',
    'labelRestricted': 'Reference 类型',
    'restrictedAuto': '自动',
    'restrictedTrue': '限制性',
    'restrictedFalse': '非限制性',
    'summaryItemLabel': '项目',
    'summaryValueLabel': '数值',
    'strongCorrelationTitle': 'Active Space',
    'labelOrbitalProcessing': '启用轨道处理',
    'labelDensityFittingAuxbasis': 'Density fitting / 辅助基组',
    'densityFittingOff': '不使用 density fitting',
    'densityFittingAuxbasisAuto': 'PySCF 自动选择辅助基组',
    'labelLocalizationMethod': '轨道局域化',
    'labelLocalizationScope': '局域化用途',
    'localizationNone': '不局域化',
    'localizationBoys': 'Boys',
    'localizationPipekMezey': 'Pipek-Mezey',
    'localizationScopeAnalysis': '结果分析',
    'localizationScopeActiveSpace': 'block2 活性空间',
    'labelUseNaturalOrbitals': '生成自然轨道占据摘要',
    'labelActiveSpace': '启用 active space',
    'labelActiveSpaceMethod': '选择方式',
    'labelActiveSpaceSolver': '活性空间求解器',
    'labelAvasTargets': 'AVAS AO/片段目标',
    'labelAvasThreshold': 'AVAS 阈值',
    'avasTargetsPlaceholder': '例如：Fe 3d 或 atom:0,1',
    'activeSpaceManual': '手动指定',
    'activeSpaceOccupationWindow': '自然占据窗口',
    'activeSpaceEnergyWindow': '能量窗口',
    'activeSpaceAvas': 'AVAS AO/片段投影',
    'labelNcas': '活性轨道数',
    'labelNelecas': '活性电子数',
    'labelActiveOrbitals': '活性轨道编号',
    'labelActiveSpaceApproved': '已人工确认 active space',
    'activeOrbitalsPlaceholder': '0-based，例如：4,5,6,7；非限制性可用 alpha:4,5; beta:4,5',
    'nelecasPlaceholder': '例如：6 或 3,3',
    'activeSpaceSummaryEmpty': '尚未选择活性空间',
    'activeSpaceScreenButton': '自动选择',
    'activeSpaceManualEditor': 'Active Space Contract',
    'orbitalBasisSection': '轨道基与排序',
    'labelOrbitalOrdering': 'DMRG 轨道排序',
    'orbitalOrderingCanonical': 'Canonical',
    'orbitalOrderingFiedler': 'Fiedler',
    'orbitalOrderingManual': '手动',
    'labelManualOrbitalOrder': '手动排列',
    'manualOrbitalOrderPlaceholder': '例如：0,2,1,3',
    'stateTargetSection': '低能态',
    'labelStateTargetNroots': '根数',
    'labelStateAverageWeights': '态平均权重',
    'stateAverageWeightsPlaceholder': '例如：0.5,0.5；留空则等权',
    'postCasSection': 'Post-CAS Correction',
    'labelScNevpt2': 'SC-NEVPT2 校正',
    'labelScNevpt2Root': 'Root',
    'labelScNevpt2DensityFit': 'Density fitting',
    'activeSpaceScreeningStarted': '开始 active-space 自动选择并生成 ActiveSpaceAudit 证据。',
    'activeSpaceApprove': '确认 Active Space',
    'activeSpaceEdit': '编辑 Active Space',
    'activeSpaceRunCasscf': '运行 CASSCF',
    'activeSpaceApprovedMessage': 'Active space 已写回表单并标记为人工确认，可以运行 CASSCF。',
    'activeSpaceEditedMessage': 'Active space 已写回表单，仍需人工确认后再运行 CASSCF。',
    'activeSpaceNoAudit': '当前结果没有可用的 ActiveSpaceAudit。',
    'activeSpaceApprovalRequired': '请先确认 Active Space，再运行 CASSCF。',
    'labelRequest': 'Message',
    'atomPlaceholder': '例如：O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
    'requestPlaceholder': '例如：水分子 B3LYP/6-31G 单点能。',
    'approvalAssumptionsTitle': '几何假设',
    'approvalAtomTitle': '待确认结构草案',
    'approvalAtomPlaceholder': '在这里核对或修改结构草案',
    'approvalActiveSpaceEvidenceTitle': 'Active Space 证据',
    'approvalActiveSpaceTitle': '待确认 Active Space',
    'approvalActiveSpacePlaceholder': '核对或修改 active_space JSON',
    'approvalActiveSpaceSystemMessage': '自动 active-space 选择生成了候选方案。请先核对下面的结构化提案。',
    'approvalActiveSpaceDirectSystemMessage': 'CASSCF/CASCI 请求包含尚未批准的 active space。请先核对下面的结构化提案。',
    'approvalActiveSpaceSource': '这版 active space 来自 ActiveSpaceAudit。',
    'approvalActiveSpacePreparedSource': '这版 active space 来自已准备但尚未批准的 CAS 请求。',
    'approvalActiveSpaceDirectSource': '这版 active space 来自被验证器暂停的 CAS 请求。',
    'approvalActiveSpaceRunsAfterConfirm': '确认后会生成 active-space 计算请求，需再点击 Start Calculation 执行。',
    'approvalActiveSpaceReviewTitle': 'Active Space 提案',
    'approvalActiveSpaceOrbitalsTitle': '选中轨道',
    'approvalActiveSpaceJsonEditor': '高级 JSON 编辑',
    'computeConfirmHint': '请求已经整理完成。你可以开始计算，或者继续修改结构与参数后再发送更新。',
    'run': '准备请求',
    'runLoading': '整理中...',
    'analyzeResult': 'Analyze Results',
    'analyzeLoading': '分析中...',
    'analyzeUnavailable': '未配置 LLM，无法分析结果',
    'refreshStatus': '刷新状态',
    'refreshingStatus': '刷新中...',
    'clear': '清空结果',
    'stop': '中止计算',
    'stopping': '正在中止...',
    'conversationTitle': 'Assistant Conversation',
    'summaryTitle': '结果总览',
    'summaryDescription': '从 task_report 中提炼结果、诊断和追踪信息。',
    'statStatusLabel': '执行状态',
    'statRetryLabel': '重试次数',
    'statAttemptsLabel': '尝试历史',
    'structuredResultsSummary': '结构化结果',
    'strongCorrelationDiagnosticsTitle': '强关联诊断',
    'strongCorrelationDiagnosticsEmpty': '未请求强关联诊断',
    'activeSpaceAuditTitle': 'Active Space 审计',
    'activeSpaceAuditEmpty': '未生成 ActiveSpaceAudit',
    'diagnosticDataAnalysisTitle': '数据分析',
    'diagnosticLevelTitle': '强关联程度判断',
    'diagnosticMethodTitle': '推荐方法',
    'llmResultAnalysisTitle': 'Analyze Results',
    'llmResultAnalysisEmpty': '尚未请求 LLM 结果分析',
    'llmResultAnalysisUnavailable': '当前未配置 LLM，无法分析结果',
    'inputPreviewSummary': 'PySCF 输入预览',
    'rawOutputSummary': '原始输出与调试文本',
    'defaultsAndRetriesTitle': '默认值与重试',
    'diagnosticsTitle': '错误与补充信息',
    'attemptsTitle': '尝试历史',
    'summaryRequestLabel': '请求摘要',
    'summaryDefaultsLabel': '默认假设',
    'summaryGeneratedLabel': '自动生成',
    'summaryPrecisionLabel': '精度预期',
    'sourceLlm': 'LLM 整理',
    'sourceStructuredSeed': '表单直出',
    'sourceFallback': '未调用 LLM，直接使用表单',
    'sourceLlmUnavailable': 'LLM 不可用',
    'sourceEmptyInput': '输入不足',
    'llmScopeUnknown': '未知',
    'llmScopeSupported': '已调用 LLM，结构化输出可用',
    'llmScopeUnsupported': '已调用 LLM，回退到文本 JSON 解析',
    'llmScopeDisabled': '未配置 LLM',
    'llmScopePending': 'LLM 状态待确认',
    'summaryApprovalPending': '存在待审批的结构草案',
    'summaryNeedsInput': '需要先补充信息后才能执行',
    'summaryEmpty': '等待输入',
    'statusNotRun': '未运行',
    'statusPrepared': '可开始计算',
    'statusNeedsInput': '需要输入',
    'statusNeedsReview': '需要审批',
    'statusChanged': '配置已更改',
    'statusCalculating': '计算中',
    'statusQueued': '排队中',
    'statusRunning': '运行中',
    'statusCompleted': '已完成',
    'statusFailed': '失败',
    'statusCancelled': '已中止',
    'preparationReadyNote': '请求已准备，尚未运行任何计算。',
    'preparationNeedsInputNote': '请先补充标记的输入，然后重新准备请求。',
    'preparationReviewNote': '请求在运行前需要人工审批。',
    'preparationChangedNote': '计算配置已更改，请重新点击“Prepare Request”。',
    'molecularStructureRequired': '请在 Molecular Structure 中输入分子坐标，或在 Message 中描述需要生成的分子结构。',
    'configurationChangedMessage': '配置已更改。请重新准备请求后再开始计算。',
    'block2SolverRequiresCas': 'block2 DMRG 仅可作为 CASCI/CASSCF 的活性空间求解器。请先选择 CASCI 或 CASSCF。',
    'block2SolverDependencyHint': '在所选的本地或远程执行环境中需要可选依赖 {dependency}：`python -m pip install {dependency}`。',
    'noDefaultsOrRetries': '暂无默认值或重试信息',
    'defaultsAndRetriesEmpty': '无默认值应用，也没有发生重试',
    'noDiagnostics': '暂无诊断信息',
    'diagnosticsEmpty': '无错误或待补充信息',
    'attemptsEmpty': '暂无尝试历史',
    'notExecutedYet': '尚未执行',
    'noPyscfOut': '无 pyscf_out 输出',
    'sameAsPyscfOut': '与 pyscf_out 相同，已省略重复显示',
    'noAgentOut': '无 agent_out 输出',
    'noStderrAnalysis': '无 raw_stderr / analysis_text 输出',
    'runPreparing': '正在整理请求，请稍候...',
    'runStarted': '任务开始执行。',
    'runCompleted': '任务执行完成。',
    'runEnded': '任务已结束。',
    'prepareFailed': '请求准备失败',
    'runFailed': '执行失败',
    'runCancelled': '计算已由用户中止。',
    'runCancellationFailed': '无法中止计算',
    'runStatusUnavailable': '当前任务还没有可查询的 JobHandle。',
    'runStatusUpdated': '当前任务状态：{state}。{message}',
    'runStatusRefreshFailed': '状态刷新失败',
    'runConnectionInterrupted': '状态轮询连接已中断；远程任务不会因此停止。请点击“刷新状态”继续查询。',
    'requestFailed': '请求失败',
    'analyzeNotConfigured': '当前未配置 LLM，无法执行结果分析。',
    'analyzeNoResult': '当前还没有可供分析的执行结果。',
    'analyzeGenerating': '正在生成正式结果分析，请稍候...',
    'analyzeFailed': '结果分析失败',
    'analyzeDone': '已生成正式结果分析。',
    'analyzeEmptyResult': '分析已生成，但返回内容为空。',
    'approvalEmptyDraft': '结构草案不能为空；请修改后再确认，或取消本次草案。',
    'approvalInvalidActiveSpace': 'Active space JSON 无法解析或缺少 ncas/nelecas/orbital_indices。',
    'approvalFailed': '审批状态更新失败',
    'approvalConfirmed': '已确认结构草案。',
    'approvalActiveSpaceConfirmed': '已确认 Active Space，并已生成 active-space 计算请求。',
    'approvalConfirmRunHint': '结构草案已确认。确认计算会立即开始执行；如果还想调整参数，可以先继续修改。',
    'approvalCanceled': '已取消这版结构草案。你可以继续补充约束，或直接手动填写结构。',
    'approvalActiveSpaceCanceled': '已取消这版 Active Space 候选。你可以重新筛选或手动填写 CAS 参数。',
    'continueEditingPrompt': '请继续补充分子结构或修改你的描述。',
    'computeNotReady': '',
    'computeReady': '',
    'computeInvalidated': '',
    'approvalGeneratedSource': '这版结构来自默认分子草案。',
    'approvalLlmSource': '这版结构由 LLM 根据你的描述整理。',
    'approvalNeedsInputAfterConfirm': '确认后仍需补充',
    'approvalRunsImmediately': '确认后会直接开始计算。',
    'diagnosticMissingField': '缺少字段',
    'confirmAndContinue': '确认并继续',
    'cancel': '取消',
    'confirmCompute': '确认计算',
    'systemLabel': '系统',
    'assistantLabel': '助手',
    'userLabel': '你',
    'outputLabels': {
      'energy': '总能',
      'homo_lumo': 'HOMO/LUMO',
      'dipole': '偶极矩',
      'strong_correlation_diagnostics': '强关联诊断',
      'band_gap': '平均场带隙',
      'fermi_energy': '费米能级',
      'band_structure': '高对称路径能带',
    },
  },
  'en': {
    'htmlLang': 'en',
    'pageTitle': 'Calculation Assistant',
    'toggleLabel': '中文',
    'heroTitle': 'Calculation Assistant',
    'heroDescription': 'Individual task calculation',
    'openStudyAgent': 'Planner Agent',
    'configTitle': 'Task Setup',
    'configDescription': '',
    'labelTaskFamily': 'System',
    'taskFamilyMolecular': 'Molecular Electronic Structure',
    'taskFamilyModelHamiltonian': 'Model Hamiltonian',
    'taskFamilyPeriodic': 'Periodic Electronic Structure',
    'taskSessionsTitle': 'Tasks',
    'newTask': 'New Task',
    'taskSessionMolecular': 'Molecular Task',
    'taskSessionModelHamiltonian': 'Model Task',
    'taskSessionPeriodic': 'Periodic Task',
    'taskStatusDraft': 'Draft',
    'taskStatusReady': 'Ready',
    'taskStatusChanged': 'Changed',
    'taskStatusRunning': 'Calculating',
    'taskStatusCancelled': 'Cancelled',
    'taskStatusCompleted': 'Completed',
    'taskStatusFailed': 'Failed',
    'taskStarted': 'Started a new task session',
    'builderInputLinked': 'The model Hamiltonian input generated by the builder is linked to the current task.',
    'workDirLockedHint': 'Locked to the actual run directory',
    'labelWorkDir': 'Work Directory',
    'labelExecutionTarget': 'Execution',
    'labelResourceProfile': 'Resource Profile',
    'workDirPlaceholder': 'Leave empty to use runs/ under the current directory',
    'effectiveWorkDirLabel': 'Using',
    'labelModelSolver': 'Model Solver',
    'previewModelHamiltonian': 'Preview Structure',
    'modelHamiltonianPreviewTitle': 'Model Structure Preview',
    'modelHamiltonianPreviewEmpty': 'Generate an input file with the builder to preview the model graph here.',
    'modelHamiltonianPreviewLoading': 'Reading model structure...',
    'modelHamiltonianPreviewFailed': 'Model structure preview failed',
    'modelHamiltonianPreviewTruncated': 'Graph preview truncated to the first sites and bonds.',
    'modelHamiltonianPreviewNoSites': 'The input file does not contain drawable sites.',
    'openModelHamiltonianBuilder': 'Open Model Hamiltonian Builder',
    'modelHamiltonianMissingFile': 'Generate a model Hamiltonian PySCF input file with the builder UI first.',
    'modelHamiltonianPrepared': 'The model Hamiltonian request is ready. You can confirm the calculation.',
    'periodicStructureTitle': 'Periodic Structure',
    'labelPeriodicFormat': 'Structure Format',
    'labelPeriodicFile': 'Read Structure File',
    'labelPeriodicStructure': 'POSCAR / CIF Content',
    'periodicStructurePlaceholder': 'Upload or paste POSCAR/CIF content',
    'previewPeriodicStructure': 'Check Structure',
    'periodicPreviewEmpty': 'Upload or paste a POSCAR/CIF file, then check the structure.',
    'periodicPreviewLoading': 'Parsing periodic structure...',
    'periodicPreviewFailed': 'Periodic structure parsing failed',
    'labelPeriodicBasis': 'Periodic Basis',
    'labelPeriodicPseudo': 'Pseudopotential',
    'labelPeriodicMethod': 'Method',
    'labelPeriodicXc': 'XC Functional',
    'labelPeriodicKmesh': 'k-point Mesh',
    'labelPeriodicKpointScheme': 'k-point Scheme',
    'labelPeriodicBandPathMode': 'Band Path',
    'periodicBandPathModeAuto': 'Auto',
    'periodicBandPathModeSeekpath': 'SeeK-path Primitive Cell',
    'periodicBandPathModeCustom': 'Custom Points',
    'periodicBandPathModeExplicit': 'Explicit Coordinates',
    'labelPeriodicBandPath': 'High-symmetry Path',
    'periodicBandPathPlaceholder': 'For example, GXWKGLUWLK,UX',
    'labelPeriodicBandSpecialPoints': 'Reduced Special-point Coordinates',
    'periodicBandSpecialPointsPlaceholder': 'G = 0, 0, 0\nX = 0.5, 0, 0.5',
    'labelPeriodicBandPathNpoints': 'Band-path Points',
    'labelPeriodicBandPathReferenceDistance': 'Path Spacing (1/Angstrom)',
    'labelPeriodicBandPathSymprec': 'Symmetry Tolerance (Angstrom)',
    'periodicNumericsTitle': 'Numerical Controls',
    'labelPeriodicKpointShift': 'k-point Shift (mesh steps)',
    'labelPeriodicPrecision': 'Integral Precision',
    'labelPeriodicKeCutoff': 'Plane-wave Auxiliary Cutoff (Hartree)',
    'labelPeriodicFftMesh': 'FFT Mesh (blank for automatic)',
    'labelPeriodicDensityFitting': 'Density Fitting',
    'labelPeriodicDfAuxbasis': 'Auxiliary Gaussian Basis (optional)',
    'periodicDfAuxbasisPlaceholder': 'GDF / MDF only; blank for automatic',
    'labelPeriodicExxdiv': 'Exact-exchange Divergence Treatment',
    'labelPeriodicSmearing': 'Occupation Smearing',
    'labelPeriodicSmearingSigma': 'Smearing Sigma (Hartree)',
    'labelPeriodicSmearingFixSpin': 'Fix alpha / beta electron counts separately',
    'labelPeriodicSpin': 'Spin (Nalpha - Nbeta)',
    'periodicMissingStructure': 'Upload or paste a POSCAR/CIF structure first.',
    'periodicPrepared': 'The periodic request is ready. You can confirm the calculation.',
    'labelAtom': 'Molecular Structure',
    'sectionStructure': 'Structure',
    'sectionCalculationSettings': 'Calculation Settings',
    'sectionSpinParameters': 'Charge and Spin',
    'sectionActiveSpace': 'Active Space',
    'labelBasis': 'Basis Set',
    'labelMethod': 'Method',
    'labelXc': 'XC Functional',
    'labelJob': 'Job Type',
    'jobSinglePoint': 'single point',
    'advancedParameters': 'Advanced Parameters',
    'labelCharge': 'Charge',
    'labelSpin': 'Spin (Nα - Nβ)',
    'labelRestricted': 'Reference Type',
    'restrictedAuto': 'Auto',
    'restrictedTrue': 'Restricted',
    'restrictedFalse': 'Unrestricted',
    'summaryItemLabel': 'Item',
    'summaryValueLabel': 'Value',
    'strongCorrelationTitle': 'Active Space',
    'labelOrbitalProcessing': 'Enable Orbital Processing',
    'labelDensityFittingAuxbasis': 'Density Fitting / Auxiliary Basis',
    'densityFittingOff': 'No density fitting',
    'densityFittingAuxbasisAuto': 'PySCF auto auxiliary basis',
    'labelLocalizationMethod': 'Orbital Localization',
    'labelLocalizationScope': 'Localization Use',
    'localizationNone': 'No localization',
    'localizationBoys': 'Boys',
    'localizationPipekMezey': 'Pipek-Mezey',
    'localizationScopeAnalysis': 'Result Analysis',
    'localizationScopeActiveSpace': 'block2 active space',
    'labelUseNaturalOrbitals': 'Summarize Natural-Orbital Occupations',
    'labelActiveSpace': 'Enable Active Space',
    'labelActiveSpaceMethod': 'Selection Mode',
    'labelActiveSpaceSolver': 'Active-space solver',
    'labelAvasTargets': 'AVAS AO/fragment targets',
    'labelAvasThreshold': 'AVAS threshold',
    'avasTargetsPlaceholder': 'e.g. Fe 3d or atom:0,1',
    'activeSpaceManual': 'Manual',
    'activeSpaceOccupationWindow': 'Occupation Window',
    'activeSpaceEnergyWindow': 'Energy Window',
    'activeSpaceAvas': 'AVAS AO/fragment projection',
    'labelNcas': 'Active Orbital Count',
    'labelNelecas': 'Active Electrons',
    'labelActiveOrbitals': 'Active Orbital Indices',
    'labelActiveSpaceApproved': 'Active space reviewed by user',
    'activeOrbitalsPlaceholder': '0-based, e.g. 4,5,6,7; unrestricted: alpha:4,5; beta:4,5',
    'nelecasPlaceholder': 'e.g. 6 or 3,3',
    'activeSpaceSummaryEmpty': 'No active space selected',
    'activeSpaceScreenButton': 'Auto Select',
    'activeSpaceManualEditor': 'Active Space Contract',
    'orbitalBasisSection': 'Orbital Basis and Ordering',
    'labelOrbitalOrdering': 'DMRG Orbital Ordering',
    'orbitalOrderingCanonical': 'Canonical',
    'orbitalOrderingFiedler': 'Fiedler',
    'orbitalOrderingManual': 'Manual',
    'labelManualOrbitalOrder': 'Manual Permutation',
    'manualOrbitalOrderPlaceholder': 'e.g. 0,2,1,3',
    'stateTargetSection': 'Low-Energy States',
    'labelStateTargetNroots': 'Number of Roots',
    'labelStateAverageWeights': 'State-average Weights',
    'stateAverageWeightsPlaceholder': 'e.g. 0.5,0.5; blank means equal weights',
    'postCasSection': 'Post-CAS Correction',
    'labelScNevpt2': 'SC-NEVPT2 correction',
    'labelScNevpt2Root': 'Root',
    'labelScNevpt2DensityFit': 'Density fitting',
    'activeSpaceScreeningStarted': 'Starting automatic active-space selection and generating ActiveSpaceAudit evidence.',
    'activeSpaceApprove': 'Approve Active Space',
    'activeSpaceEdit': 'Edit Active Space',
    'activeSpaceRunCasscf': 'Run CASSCF',
    'activeSpaceApprovedMessage': 'Active space was written back to the form and marked as user-approved. CASSCF is ready.',
    'activeSpaceEditedMessage': 'Active space was written back to the form. Review and approve it before CASSCF.',
    'activeSpaceNoAudit': 'No ActiveSpaceAudit is available in the current result.',
    'activeSpaceApprovalRequired': 'Approve the active space before running CASSCF.',
    'labelRequest': 'Message',
    'atomPlaceholder': 'Example: O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
    'requestPlaceholder': 'Example: Water single-point with B3LYP/6-31G.',
    'approvalAssumptionsTitle': 'Geometry Assumptions',
    'approvalAtomTitle': 'Structure Draft To Review',
    'approvalAtomPlaceholder': 'Review or edit the structure draft here',
    'approvalActiveSpaceEvidenceTitle': 'Active Space Evidence',
    'approvalActiveSpaceTitle': 'Active Space To Review',
    'approvalActiveSpacePlaceholder': 'Review or edit the active_space JSON',
    'approvalActiveSpaceSystemMessage': 'Automatic active-space selection generated a candidate. Review the structured proposal below.',
    'approvalActiveSpaceDirectSystemMessage': 'The CASSCF/CASCI request contains an unapproved active space. Review the structured proposal below.',
    'approvalActiveSpaceSource': 'This active space came from ActiveSpaceAudit.',
    'approvalActiveSpacePreparedSource': 'This active space came from the prepared, unapproved CAS request.',
    'approvalActiveSpaceDirectSource': 'This active space came from the CAS request paused by validation.',
    'approvalActiveSpaceRunsAfterConfirm': 'Confirmation will prepare an active-space request; click Start Calculation to run it.',
    'approvalActiveSpaceReviewTitle': 'Active Space Proposal',
    'approvalActiveSpaceOrbitalsTitle': 'Selected Orbitals',
    'approvalActiveSpaceJsonEditor': 'Advanced JSON Edit',
    'computeConfirmHint': 'The request is prepared. You can start the calculation now, or keep editing the structure and parameters before sending an update.',
    'run': 'Prepare Request',
    'runLoading': 'Preparing...',
    'analyzeResult': 'Analyze Results',
    'analyzeLoading': 'Analyzing...',
    'analyzeUnavailable': 'LLM is not configured',
    'refreshStatus': 'Status',
    'refreshingStatus': 'Refreshing...',
    'clear': 'Clear',
    'stop': 'Stop',
    'stopping': 'Stopping...',
    'conversationTitle': 'Assistant Conversation',
    'summaryTitle': 'Result Overview',
    'summaryDescription': 'A compact view of execution results, diagnostics, and trace data from task_report.',
    'statStatusLabel': 'Execution Status',
    'statRetryLabel': 'Retry Count',
    'statAttemptsLabel': 'Attempts',
    'structuredResultsSummary': 'Structured Results',
    'strongCorrelationDiagnosticsTitle': 'Strong Correlation Diagnostics',
    'strongCorrelationDiagnosticsEmpty': 'Strong correlation diagnostics were not requested',
    'activeSpaceAuditTitle': 'Active Space Audit',
    'activeSpaceAuditEmpty': 'ActiveSpaceAudit was not generated',
    'diagnosticDataAnalysisTitle': 'Data Analysis',
    'diagnosticLevelTitle': 'Correlation Level',
    'diagnosticMethodTitle': 'Method Recommendation',
    'llmResultAnalysisTitle': 'Analyze Results',
    'llmResultAnalysisEmpty': 'LLM result analysis has not been requested yet',
    'llmResultAnalysisUnavailable': 'LLM is not configured for result analysis',
    'inputPreviewSummary': 'PySCF Input Preview',
    'rawOutputSummary': 'Raw Output And Debug Text',
    'defaultsAndRetriesTitle': 'Defaults And Retries',
    'diagnosticsTitle': 'Errors And Missing Info',
    'attemptsTitle': 'Attempt History',
    'summaryRequestLabel': 'Request Summary',
    'summaryDefaultsLabel': 'Defaults',
    'summaryGeneratedLabel': 'Auto-generated',
    'summaryPrecisionLabel': 'Precision Note',
    'sourceLlm': 'LLM parsed request',
    'sourceStructuredSeed': 'Structured form request',
    'sourceFallback': 'LLM not used; form submitted directly',
    'sourceLlmUnavailable': 'LLM unavailable',
    'sourceEmptyInput': 'Input incomplete',
    'llmScopeUnknown': 'Unknown',
    'llmScopeSupported': 'LLM called; structured output available',
    'llmScopeUnsupported': 'LLM called; fell back to text JSON parsing',
    'llmScopeDisabled': 'LLM not configured',
    'llmScopePending': 'LLM status pending',
    'summaryApprovalPending': 'A structure draft is waiting for approval',
    'summaryNeedsInput': 'More information is required before execution',
    'summaryEmpty': 'Waiting for input',
    'statusNotRun': 'Not run',
    'statusPrepared': 'Ready to run',
    'statusNeedsInput': 'Needs input',
    'statusNeedsReview': 'Review required',
    'statusChanged': 'Configuration changed',
    'statusCalculating': 'Calculating',
    'statusQueued': 'Queued',
    'statusRunning': 'Running',
    'statusCompleted': 'Completed',
    'statusFailed': 'Failed',
    'statusCancelled': 'Cancelled',
    'preparationReadyNote': 'The request is prepared. No calculation has been run yet.',
    'preparationNeedsInputNote': 'Complete the highlighted input before preparing the request again.',
    'preparationReviewNote': 'The request requires human approval before it can run.',
    'preparationChangedNote': 'The calculation configuration changed. Prepare the request again before running it.',
    'molecularStructureRequired': 'Enter coordinates in Molecular Structure, or describe the molecular structure to generate in Message.',
    'configurationChangedMessage': 'The configuration changed. Prepare the request again before starting the calculation.',
    'block2SolverRequiresCas': 'block2 DMRG is available only as an active-space solver for CASCI or CASSCF. Select CASCI or CASSCF first.',
    'block2SolverDependencyHint': 'The selected local or remote execution environment must provide the optional {dependency} package: `python -m pip install {dependency}`.',
    'noDefaultsOrRetries': 'No defaults were applied and no retries were needed',
    'defaultsAndRetriesEmpty': 'No defaults were applied and no retries were needed',
    'noDiagnostics': 'No diagnostics',
    'diagnosticsEmpty': 'No errors or missing information',
    'attemptsEmpty': 'No attempts yet',
    'notExecutedYet': 'Not executed yet',
    'noPyscfOut': 'No pyscf_out output',
    'sameAsPyscfOut': 'Same as pyscf_out; duplicate display omitted',
    'noAgentOut': 'No agent_out output',
    'noStderrAnalysis': 'No raw_stderr / analysis_text output',
    'runPreparing': 'Preparing the request. Please wait...',
    'runStarted': 'The task has started running.',
    'runCompleted': 'The task completed successfully.',
    'runEnded': 'The task has finished.',
    'prepareFailed': 'Request preparation failed',
    'runFailed': 'Execution failed',
    'runCancelled': 'The calculation was stopped by the user.',
    'runCancellationFailed': 'The calculation could not be stopped',
    'runStatusUnavailable': 'The current task does not have a JobHandle to inspect.',
    'runStatusUpdated': 'Current task status: {state}. {message}',
    'runStatusRefreshFailed': 'Status refresh failed',
    'runConnectionInterrupted': 'Status polling was interrupted; the remote task was not stopped. Click Status to query it again.',
    'requestFailed': 'Request failed',
    'analyzeNotConfigured': 'LLM is not configured for result analysis.',
    'analyzeNoResult': 'There is no execution result to analyze yet.',
    'analyzeGenerating': 'Generating the formal result analysis. Please wait...',
    'analyzeFailed': 'Analysis failed',
    'analyzeDone': 'The formal result analysis is ready.',
    'analyzeEmptyResult': 'Analysis finished, but the returned content was empty.',
    'approvalEmptyDraft': 'The structure draft cannot be empty. Edit it before confirming, or cancel this draft.',
    'approvalInvalidActiveSpace': 'The active-space JSON could not be parsed or is missing ncas/nelecas/orbital_indices.',
    'approvalFailed': 'Approval state update failed',
    'approvalConfirmed': 'The structure draft has been confirmed.',
    'approvalActiveSpaceConfirmed': 'The active space has been approved and an active-space request is ready.',
    'approvalConfirmRunHint': 'The structure draft is confirmed. Confirm calculation to run immediately, or keep editing parameters first.',
    'approvalCanceled': 'This structure draft was canceled. You can keep adding constraints or fill in the structure manually.',
    'approvalActiveSpaceCanceled': 'This active-space candidate was canceled. You can screen again or fill CAS parameters manually.',
    'continueEditingPrompt': 'Please continue by providing the molecular structure or refining your request.',
    'computeNotReady': '',
    'computeReady': '',
    'computeInvalidated': '',
    'approvalGeneratedSource': 'This structure came from the default molecule draft.',
    'approvalLlmSource': 'This structure was drafted by the LLM from your description.',
    'approvalNeedsInputAfterConfirm': 'Still required after confirmation',
    'approvalRunsImmediately': 'The calculation will start immediately after confirmation.',
    'diagnosticMissingField': 'Missing field',
    'confirmAndContinue': 'Confirm And Continue',
    'cancel': 'Cancel',
    'confirmCompute': 'Start Calculation',
    'systemLabel': 'System',
    'assistantLabel': 'Assistant',
    'userLabel': 'You',
    'outputLabels': {
      'energy': 'Energy',
      'homo_lumo': 'HOMO/LUMO',
      'dipole': 'Dipole',
      'strong_correlation_diagnostics': 'Strong correlation diagnostics',
      'band_gap': 'Mean-field band gap',
      'fermi_energy': 'Fermi level',
      'band_structure': 'High-symmetry band structure',
    },
  },
}


def _build_options(options: Sequence[Any], selected: str) -> str:
    rendered = []
    for option in options:
        label_key = None
        if isinstance(option, tuple) and len(option) >= 3:
            value, label, label_key = option[:3]
        elif isinstance(option, tuple):
            value, label = option
        else:
            value = label = option
        attrs = ' selected' if value == selected else ''
        if label_key:
            attrs += ' data-label-key="{0}"'.format(html.escape(str(label_key), quote=True))
        rendered.append(
            '<option value="{value}"{attrs}>{label}</option>'.format(
                value=html.escape(str(value)),
                attrs=attrs,
                label=html.escape(str(label)),
            )
        )
    return '\n'.join(rendered)


def _build_output_options(selected_outputs: Sequence[str], output_options: Sequence[tuple]) -> str:
    rendered = []
    selected = set(selected_outputs or [])
    if not selected:
        selected = {'energy'}
    for value, label in output_options:
        checked = ' checked' if value in selected else ''
        rendered.append(
            '<label class="checkbox-item"><input type="checkbox" name="outputs" value="{value}"{checked}>{label}</label>'.format(
                value=html.escape(value),
                checked=checked,
                label=html.escape(label),
            )
        )
    return '\n'.join(rendered)


@versioned_assets
def build_index_html(*, llm_request_builder_module: Any = None) -> str:
  try:
    example = json.loads(example_request())
  except (TypeError, json.JSONDecodeError):
    LOGGER.warning(
      'Failed to parse example_request JSON; using empty defaults',
      exc_info=True,
    )
    example = {}

  llm_cache_scope = 'llm:unknown'
  if llm_request_builder_module is not None and hasattr(llm_request_builder_module, 'get_llm_cache_scope'):
    llm_cache_scope = llm_request_builder_module.get_llm_cache_scope()

  block2_capability = default_registry().capability(
    'block2_dmrg', namespace='molecular.active_space_solver'
  )
  block2_dependency = (
    block2_capability.metadata.get('optional_dependency', 'block2')
    if block2_capability is not None
    else 'block2'
  )

  return HTML_PAGE.format(
    default_atom='',
    default_work_dir_json=json.dumps(str(resolve_work_dir()), ensure_ascii=False),
    default_request=html.escape(example.get('request', '')),
    task_family_options=_build_options(TASK_FAMILY_OPTIONS, example.get('task_type', 'molecular')),
    basis_options=_build_options(BASIS_OPTIONS, example.get('basis', BASIS_OPTIONS[0])),
    auxbasis_options=_build_options(AUXBASIS_OPTIONS, '__off__'),
    auxbasis_recommendations_json=json.dumps(DENSITY_FITTING_AUXBASIS_RECOMMENDATIONS, ensure_ascii=False),
    method_options=_build_options(METHOD_OPTIONS, example.get('method', METHOD_OPTIONS[0][0])),
    job_options=_build_options(JOB_OPTIONS, example.get('job', 'single_point')),
    xc_options=_build_options(XC_OPTIONS, example.get('xc', XC_OPTIONS[0])),
    model_solver_options=_build_options(MODEL_SOLVER_OPTIONS, 'fci'),
    model_dmet_reference_density_options=_build_options(
        DMET_REFERENCE_DENSITY_OPTIONS,
        'pm',
    ),
    periodic_method_options=_build_options(PERIODIC_METHOD_OPTIONS, 'dft'),
    periodic_correlation_options=_build_options(PERIODIC_CORRELATION_OPTIONS, 'none'),
    periodic_dmft_impurity_solver_options=_build_options(
        PERIODIC_DMFT_IMPURITY_SOLVER_OPTIONS,
        'cc',
    ),
    periodic_dmft_localization_options=_build_options(
        PERIODIC_DMFT_LOCALIZATION_OPTIONS,
        'iao',
    ),
    periodic_basis_options=_build_options(PERIODIC_BASIS_OPTIONS, DEFAULT_PERIODIC_BASIS_SET),
    periodic_pseudo_options=_build_options(PERIODIC_PSEUDO_OPTIONS, 'gth-pbe'),
    periodic_xc_options=_build_options(PERIODIC_XC_OPTIONS, 'pbe'),
    periodic_density_fitting_options=_build_options(PERIODIC_DENSITY_FITTING_OPTIONS, 'fft'),
    periodic_kpoint_scheme_options=_build_options(PERIODIC_KPOINT_SCHEME_OPTIONS, 'gamma_centered'),
    periodic_band_path_mode_options=_build_options(PERIODIC_BAND_PATH_MODE_OPTIONS, 'auto'),
    periodic_smearing_options=_build_options(PERIODIC_SMEARING_OPTIONS, 'none'),
    periodic_exxdiv_options=_build_options(PERIODIC_EXXDIV_OPTIONS, 'ewald'),
    localization_method_options=_build_options(LOCALIZATION_METHOD_OPTIONS, 'none'),
    active_space_method_options=_build_options(ACTIVE_SPACE_METHOD_OPTIONS, 'manual'),
    active_space_solver_options=_build_options(ACTIVE_SPACE_SOLVER_OPTIONS, 'fci'),
    block2_dependency=html.escape(str(block2_dependency)),
    output_options=_build_output_options(example.get('outputs', []), OUTPUT_OPTIONS),
    model_output_options=_build_output_options([], MODEL_OUTPUT_OPTIONS),
    periodic_output_options=_build_output_options(['energy', 'band_gap', 'fermi_energy'], PERIODIC_OUTPUT_OPTIONS),
    default_molecular_outputs_json=json.dumps(list(DEFAULT_ANALYSIS), ensure_ascii=False),
    default_periodic_outputs_json=json.dumps(['energy', 'band_gap', 'fermi_energy'], ensure_ascii=False),
    default_periodic_basis_json=json.dumps(DEFAULT_PERIODIC_BASIS_SET, ensure_ascii=False),
    translations_json=json.dumps(UI_TRANSLATIONS, ensure_ascii=False),
    llm_cache_scope=json.dumps(llm_cache_scope, ensure_ascii=False),
  )
