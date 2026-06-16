// gallery selection + bulk decisions
function selected() {
  return Array.from(document.querySelectorAll('.tile')).filter(t =>
    t.querySelector('.pick') && t.querySelector('.pick').checked).map(t => t.dataset.id);
}
function refreshSel() {
  const n = selected().length;
  const el = document.getElementById('selcount');
  if (el) el.textContent = n + ' 选中';
  document.querySelectorAll('.tile').forEach(t => {
    const p = t.querySelector('.pick');
    t.classList.toggle('sel', p && p.checked);
  });
}
document.addEventListener('change', e => {
  if (e.target.id === 'selall') {
    document.querySelectorAll('.tile .pick').forEach(p => p.checked = e.target.checked);
    refreshSel();
  } else if (e.target.classList && e.target.classList.contains('pick')) {
    refreshSel();
  }
});
async function bulk(decision) {
  const ids = selected();
  if (!ids.length) { alert('未选中'); return; }
  if (!confirm(`对 ${ids.length} 条执行 ${decision}?`)) return;
  await fetch('/api/bulk_decision', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({asset_ids: ids, decision, reviewer: 'human'})
  });
  location.reload();
}

// bulk over the ENTIRE current filter (all pages)
async function bulkFilter(decision) {
  const n = window.QA_TOTAL || 0;
  if (!n) { alert('无匹配'); return; }
  if (!confirm(`对整个筛选结果 ${n} 条执行 ${decision}？此操作可被后续人工覆盖。`)) return;
  const r = await fetch('/api/bulk_by_filter', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filters: window.QA_FILTERS, decision, reviewer: 'human'})
  });
  const j = await r.json();
  alert(`已处理 ${j.n} 条`);
  location.reload();
}

// human override of one questionnaire item on the detail page
async function qaOverride(q, item, answer) {
  const box = document.querySelector('.sticky-actions');
  const id = box ? box.dataset.id : null;
  if (!id) return;
  await fetch('/api/qa_override', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({asset_id: id, questionnaire: q, item, answer, reviewer: 'human'})
  });
  location.reload();
}

// asset-detail decision
async function decide(decision) {
  const box = document.querySelector('.sticky-actions');
  if (!box) return;
  const id = box.dataset.id, next = box.dataset.next;
  const reason = (document.getElementById('reason') || {}).value || '';
  await fetch('/api/decision', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({asset_id: id, decision, reason, reviewer: 'human'})
  });
  if (next) location.href = '/asset/' + next; else history.back();
}
// keyboard shortcuts on detail page
document.addEventListener('keydown', e => {
  if (!document.querySelector('.sticky-actions')) return;
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'k') decide('keep');
  else if (e.key === 'd') decide('drop');
  else if (e.key === 'h') decide('hold');
});
