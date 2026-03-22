const API_BASE = '/api';

/**
 * POST /generate — Gửi 2 ảnh + prompt => nhận task_id
 */
export async function generateHair(faceFile, hairFile, prompt, hairColor, colorIntensity) {
  const form = new FormData();
  form.append('face_image', faceFile);
  form.append('hair_image', hairFile);
  form.append('description', prompt || 'high quality realistic hair');

  if (hairColor && hairColor !== 'none') {
    form.append('hair_color', hairColor);
    form.append('color_intensity', String(colorIntensity ?? 0.7));
  }

  const res = await fetch(`${API_BASE}/generate`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Generate failed: ${res.status}`);
  return res.json(); // { task_id, status, message }
}

/**
 * POST /detect-faces — Gửi 1 ảnh => nhận task_id
 */
export async function detectFaces(imageFile) {
  const form = new FormData();
  form.append('image', imageFile);

  const res = await fetch(`${API_BASE}/detect-faces`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Detect faces failed: ${res.status}`);
  return res.json(); // { task_id, status, message }
}

/**
 * GET /status/{taskId} — Poll trạng thái task
 */
export async function getTaskStatus(taskId) {
  const res = await fetch(`${API_BASE}/status/${taskId}`);
  if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
  return res.json(); // { task_id, status, result_url?, faces?, error? }
}

/**
 * GET /colors — Lấy danh sách preset màu
 */
export async function getColors() {
  const res = await fetch(`${API_BASE}/colors`);
  if (!res.ok) throw new Error(`Get colors failed: ${res.status}`);
  return res.json();
}

/**
 * GET /random-pair — Lấy 2 ảnh FFHQ ngẫu nhiên
 */
export async function getRandomPair() {
  const res = await fetch(`${API_BASE}/random-pair`);
  if (!res.ok) throw new Error(`Random pair failed: ${res.status}`);
  return res.json(); // { target_url, hair_url }
}

/**
 * Poll task cho đến khi SUCCESS/FAILURE — trả về kết quả cuối
 */
export function pollTask(taskId, onUpdate, intervalMs = 2000) {
  let stopped = false;

  const poll = async () => {
    while (!stopped) {
      try {
        const data = await getTaskStatus(taskId);
        onUpdate(data);

        if (data.status === 'SUCCESS' || data.status === 'FAILURE') {
          return data;
        }
      } catch (err) {
        onUpdate({ status: 'FAILURE', error: err.message });
        return null;
      }
      await new Promise((r) => setTimeout(r, intervalMs));
    }
  };

  const promise = poll();
  return {
    promise,
    stop: () => { stopped = true; },
  };
}
