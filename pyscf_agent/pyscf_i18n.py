from __future__ import annotations

from typing import Any, Dict, Iterable


SUPPORTED_LOCALES = ('zh', 'en')


MESSAGES: Dict[str, Dict[str, str]] = {
    'zh': {
        'intent_parsed': '已解析用户请求',
        'question_missing_atom': '请提供分子结构，例如 "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"',
        'question_missing_basis': '请提供基组，例如 sto-3g、6-31g 或 cc-pvdz',
        'question_atom_format': '请使用 "Element x y z" 的格式描述每个原子，并用分号或换行分隔',
        'question_basis_format': '请检查基组名称是否只包含字母、数字及常见符号，例如 def2-svp 或 6-31g*',
        'question_unsupported_method': '目前支持 HF、DFT、MP2、CCSD、CCSD(T) 和 Full CI，请指定 method=hf/dft/mp2/ccsd/ccsd_t/fci',
        'question_missing_xc': '请提供 DFT 泛函，例如 b3lyp 或 pbe',
        'question_xc_format': '请检查 xc 泛函名称格式，例如 b3lyp、pbe0 或 cam-b3lyp',
        'question_supported_outputs': '目前仅支持以下分析输出：{outputs}',
        'question_restricted_open_shell': '开放壳层体系请设置 restricted=false，或将 spin 调整为 0',
        'validation_missing_atom': 'system.atom 中缺少分子几何结构。',
        'validation_missing_basis': 'system.basis 中缺少基组。',
        'validation_invalid_atom_type': '任务被阻塞，因为 atom 坐标必须写成类似 "C 1.3950 0.0000 0.0000" 的扁平字符串，而不是 Python 列表或元组。请将 atom 区块改成这种文本格式，并用分号或换行分隔后重新提交。',
        'validation_invalid_atom_string': 'system.atom 必须是非空字符串。',
        'validation_invalid_atom_empty': 'system.atom 至少要包含一个原子条目。',
        'validation_invalid_atom_entry': 'system.atom 的每个条目都必须写成 "Element x y z"。',
        'validation_invalid_atom_numeric': 'system.atom 中的坐标必须是数值。',
        'validation_invalid_basis_format': 'system.basis 必须是非空字符串。',
        'validation_invalid_basis_chars': 'system.basis 包含不支持的字符。',
        'validation_unsupported_method': '不支持的方法 "{method}"。当前支持的方法有：{supported_methods}。',
        'validation_missing_xc': 'DFT 计算需要提供 xc 泛函。',
        'validation_invalid_xc_format': 'method.xc 在提供时必须是非空字符串。',
        'validation_invalid_xc_chars': 'method.xc 包含不支持的字符。',
        'validation_unsupported_job': '不支持的任务类型 "{job}"。当前支持的任务类型有：{supported_jobs}。',
        'validation_unsupported_outputs': '不支持的分析输出：{unsupported_outputs}。',
        'validation_invalid_spin': 'system.spin 必须大于或等于 0。',
        'validation_invalid_max_cycle': 'runtime.max_cycle 必须是正整数。',
        'validation_invalid_conv_tol': 'runtime.conv_tol 在提供时必须是正数。',
        'validation_invalid_verbose': 'runtime.verbose 必须大于或等于 0。',
        'validation_restricted_open_shell': '限制性计算当前要求 system.spin == 0。',
        'summary_blocked_prefix': '任务未执行：{details}',
        'summary_blocked_invalid': '任务未执行：输入校验未通过',
        'summary_failed': '任务失败：{message}',
        'summary_failed_unknown': '执行过程中发生未知错误',
        'summary_unconverged_intro': '任务完成但SCF未收敛，以下结果仅供参考',
        'summary_solver_unconverged_intro': '任务完成但{solver}未达到收敛标准，以下结果仅供参考',
        'summary_current_energy': '当前能量={value:.12f} Ha',
        'summary_total_energy': '总能={value:.12f} Ha',
        'summary_homo_lumo_gap': 'HOMO={homo:.6f} Ha, LUMO={lumo:.6f} Ha, 能隙={gap:.6f} Ha',
        'summary_dipole': '偶极矩(Debye)=[{value}]',
        'summary_unknown_status': '任务状态未知：{status}',
        'summary_scf_converged': 'SCF收敛={value}',
        'yes': '是',
        'no': '否',
        'summary_success_default': '任务成功完成',
        'summary_separator': '；',
        'retry_message': '任务将按保守参数重试',
        'prepare_empty_input': '请先提供计算目标，或填写分子结构与基组等基本信息。',
        'prepare_fallback': '未配置 LLM，已直接使用表单与补充说明生成请求。',
        'prepare_unavailable': '当前未配置 LLM，无法仅凭自然语言整理执行请求。',
        'prepare_needs_more_info': '仍需补充部分信息后才能执行。',
        'prepare_assumption_summary': '当前结构草案基于标准几何假设自动生成。',
        'prepare_approval_system_message': '我先整理出了一版可编辑的结构草案。请先核对分子名称、电荷、自旋等约束；确认后我再把它写入执行请求并继续计算，取消则不会采用这版结构。',
        'prepare_active_space_approval_system_message': '{method} 请求包含待确认的 CAS({nelecas}e, {ncas}o) active space。请在执行前核对轨道、电子数和求解器设置。',
        'prepare_active_space_confirm_label': '批准 Active Space',
        'prepare_confirm_label': '确认并继续',
        'prepare_cancel_label': '取消',
        'prepare_detected_structure_question': '我可以先按标准几何起草{label}结构草案；如果同意，请直接回复“由你来生成”或“全部由你来生成”，也可以直接粘贴坐标。',
        'prepare_generation_constraints': '如果你希望我先起草结构草案，请补充分子名称、电荷、自旋等必要约束。',
        'prepare_modification_review': '我先不给你直接落这些修改。建议修改如下：{changes}。如果你同意，我再按这些修改继续执行；如果不同意，请直接说明要保留或改成什么。',
        'prepare_supported_method': '当前支持 hf、dft、mp2、ccsd、ccsd_t 和 fci。你想使用哪一种方法？',
        'prepare_supported_job': '当前支持 single_point 和 molecular_dynamics。请选择其中一种任务类型。',
        'prepare_supported_outputs': '你请求了当前不支持的分析输出：{unsupported_outputs}。目前只支持 {supported_outputs}，请确认要保留哪些输出。',
        'prepare_missing_xc': 'DFT 计算需要提供 xc 泛函，例如 b3lyp 或 pbe0。',
        'response_language': '中文',
    },
    'en': {
        'intent_parsed': 'Parsed user request',
        'question_missing_atom': 'Please provide the molecular structure, for example "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"',
        'question_missing_basis': 'Please provide a basis set, for example sto-3g, 6-31g, or cc-pvdz',
        'question_atom_format': 'Please describe each atom as "Element x y z", separated by semicolons or new lines',
        'question_basis_format': 'Please check that the basis-set name contains only letters, numbers, and common symbols, for example def2-svp or 6-31g*',
        'question_unsupported_method': 'Supported methods are HF, DFT, MP2, CCSD, CCSD(T), and Full CI. Please set method=hf/dft/mp2/ccsd/ccsd_t/fci',
        'question_missing_xc': 'Please provide a DFT xc functional, for example b3lyp or pbe',
        'question_xc_format': 'Please check the xc functional name, for example b3lyp, pbe0, or cam-b3lyp',
        'question_supported_outputs': 'Only the following analysis outputs are currently supported: {outputs}',
        'question_restricted_open_shell': 'For open-shell systems, set restricted=false or change spin to 0',
        'validation_missing_atom': 'Missing molecular geometry in system.atom.',
        'validation_missing_basis': 'Missing basis set in system.basis.',
        'validation_invalid_atom_type': 'The task was blocked because the atom coordinates must be specified as strings like "C 1.3950 0.0000 0.0000" and not as Python list/tuple values. Please resubmit the atom block in that flat text format, separated by semicolons or new lines.',
        'validation_invalid_atom_string': 'system.atom must be a non-empty string.',
        'validation_invalid_atom_empty': 'system.atom must contain at least one atom entry.',
        'validation_invalid_atom_entry': 'system.atom entries must look like "Element x y z".',
        'validation_invalid_atom_numeric': 'system.atom coordinates must be numeric.',
        'validation_invalid_basis_format': 'system.basis must be a non-empty string.',
        'validation_invalid_basis_chars': 'system.basis contains unsupported characters.',
        'validation_unsupported_method': 'Unsupported method "{method}". Supported methods: {supported_methods}.',
        'validation_missing_xc': 'DFT calculations require an xc functional.',
        'validation_invalid_xc_format': 'method.xc must be a non-empty string when provided.',
        'validation_invalid_xc_chars': 'method.xc contains unsupported characters.',
        'validation_unsupported_job': 'Unsupported job "{job}". Supported jobs: {supported_jobs}.',
        'validation_unsupported_outputs': 'Unsupported analysis outputs: {unsupported_outputs}.',
        'validation_invalid_spin': 'system.spin must be greater than or equal to 0.',
        'validation_invalid_max_cycle': 'runtime.max_cycle must be a positive integer.',
        'validation_invalid_conv_tol': 'runtime.conv_tol must be a positive number when provided.',
        'validation_invalid_verbose': 'runtime.verbose must be greater than or equal to 0.',
        'validation_restricted_open_shell': 'Restricted calculations currently require system.spin == 0.',
        'summary_blocked_prefix': 'Task was not executed: {details}',
        'summary_blocked_invalid': 'Task was not executed: input validation failed',
        'summary_failed': 'Task failed: {message}',
        'summary_failed_unknown': 'An unknown error occurred during execution',
        'summary_unconverged_intro': 'The task finished but SCF did not converge. The following results are for reference only',
        'summary_solver_unconverged_intro': 'The task finished but {solver} did not meet its convergence criteria. The following results are for reference only',
        'summary_current_energy': 'Current energy={value:.12f} Ha',
        'summary_total_energy': 'Total energy={value:.12f} Ha',
        'summary_homo_lumo_gap': 'HOMO={homo:.6f} Ha, LUMO={lumo:.6f} Ha, gap={gap:.6f} Ha',
        'summary_dipole': 'Dipole (Debye)=[{value}]',
        'summary_unknown_status': 'Unknown task status: {status}',
        'summary_scf_converged': 'SCF converged={value}',
        'yes': 'yes',
        'no': 'no',
        'summary_success_default': 'Task completed successfully',
        'summary_separator': '; ',
        'retry_message': 'The task will be retried with conservative parameters',
        'prepare_empty_input': 'Please provide the calculation goal, or fill in basic information such as structure and basis set first.',
        'prepare_fallback': 'LLM is not configured, so the request was built directly from the form and notes.',
        'prepare_unavailable': 'LLM is not configured, so natural-language-only request preparation is unavailable.',
        'prepare_needs_more_info': 'More information is required before execution.',
        'prepare_assumption_summary': 'The current structure draft was generated from standard-geometry assumptions.',
        'prepare_approval_system_message': 'I drafted an editable structure for you first. Please review constraints such as molecular identity, charge, and spin. After confirmation, I will write it into the execution request and continue the calculation.',
        'prepare_active_space_approval_system_message': 'The {method} request contains an unapproved CAS({nelecas}e, {ncas}o) active space. Review its orbitals, electron count, and solver settings before execution.',
        'prepare_active_space_confirm_label': 'Approve Active Space',
        'prepare_confirm_label': 'Confirm and continue',
        'prepare_cancel_label': 'Cancel',
        'prepare_detected_structure_question': 'I can draft a standard-geometry structure for {label} first. If that works for you, reply with "generate it" or "generate everything", or paste coordinates directly.',
        'prepare_generation_constraints': 'If you want me to draft a structure first, please provide the molecular identity, charge, spin, and any key constraints.',
        'prepare_modification_review': 'I will not apply these changes automatically yet. Proposed changes: {changes}. If you agree, I will continue with them; otherwise, tell me what to keep or change.',
        'prepare_supported_method': 'Supported methods are hf, dft, mp2, ccsd, ccsd_t, and fci. Which one do you want to use?',
        'prepare_supported_job': 'Supported jobs are single_point and molecular_dynamics. Please choose one of them.',
        'prepare_supported_outputs': 'You requested unsupported analysis outputs: {unsupported_outputs}. Currently supported outputs are {supported_outputs}. Please confirm which outputs to keep.',
        'prepare_missing_xc': 'DFT calculations require an xc functional, for example b3lyp or pbe0.',
        'response_language': 'English',
    },
}


def normalize_locale(locale: Any) -> str:
    if isinstance(locale, str):
        normalized = locale.strip().lower()
        if normalized.startswith('zh'):
            return 'zh'
    return 'en'


def t(locale: Any, key: str, **kwargs: Any) -> str:
    normalized_locale = normalize_locale(locale)
    template = MESSAGES.get(normalized_locale, MESSAGES['en']).get(key, key)
    return template.format(**kwargs)


def yes_no(locale: Any, value: bool) -> str:
    return t(locale, 'yes' if value else 'no')


def join_items(locale: Any, items: Iterable[str]) -> str:
    separator = t(locale, 'summary_separator')
    return separator.join(items)
