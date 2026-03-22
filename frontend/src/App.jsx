import { useState, useCallback, useRef } from 'react';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import DrawButton from './components/DrawButton';
import ResultPanel from './components/ResultPanel';
import PromptInput from './components/PromptInput';
import ColorPicker from './components/ColorPicker';
import FaceSelector from './components/FaceSelector';
import { generateHair, detectFaces, pollTask, getRandomPair } from './api/hairApi';

export default function App() {
  // Images (File objects hoặc URL string)
  const [faceImage, setFaceImage] = useState(null);
  const [hairImage, setHairImage] = useState(null);

  // Prompt + Color
  const [prompt, setPrompt] = useState('high quality, realistic hairstyle');
  const [selectedColor, setSelectedColor] = useState('none');
  const [colorIntensity, setColorIntensity] = useState(0.7);

  // Pipeline state
  const [loading, setLoading] = useState(false);
  const [resultUrl, setResultUrl] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState('');
  const [pipelineError, setPipelineError] = useState('');

  // Face Selection
  const [faceSelectData, setFaceSelectData] = useState(null); // { faces, title, source }
  const [selectedFaceUrl, setSelectedFaceUrl] = useState(null);
  const [selectedHairUrl, setSelectedHairUrl] = useState(null);

  // Toast
  const [toast, setToast] = useState(null);
  const toastTimer = useRef(null);

  const showToast = useCallback((message, type = 'error') => {
    setToast({ message, type });
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 4000);
  }, []);

  // ========== Face Detection Flow ==========
  const detectAndSelect = async (imageFile, source) => {
    // POST /detect-faces → poll → faces[]
    const { task_id } = await detectFaces(imageFile);

    return new Promise((resolve, reject) => {
      const { promise } = pollTask(task_id, (data) => {
        if (data.status === 'PROCESSING') {
          setPipelineStatus(source === 'face' ? 'Đang quét khuôn mặt...' : 'Đang quét ảnh tóc...');
        }
      });

      promise.then((result) => {
        if (!result || result.status === 'FAILURE') {
          reject(new Error(result?.error || 'Phát hiện khuôn mặt thất bại'));
          return;
        }

        const faces = result.faces || [];
        if (faces.length > 1) {
          // Nhiều mặt → hiện popup chọn
          resolve({ needSelect: true, faces });
        } else {
          // 0 hoặc 1 mặt → dùng luôn file gốc
          resolve({ needSelect: false, faces });
        }
      });
    });
  };

  // ========== Main Pipeline ==========
  const handleDraw = async () => {
    // Validate
    if (!faceImage) {
      showToast('Vui lòng tải lên ảnh chân dung!');
      return;
    }
    if (!hairImage) {
      showToast('Vui lòng tải lên ảnh kiểu tóc tham khảo!');
      return;
    }

    setLoading(true);
    setResultUrl(null);
    setPipelineError('');
    setPipelineStatus('Đang bắt đầu...');

    try {
      // File object cho face — dùng gốc hoặc sẽ bị thay bởi face crop
      let faceFile = faceImage instanceof File ? faceImage : null;
      let hairFile = hairImage instanceof File ? hairImage : null;

      // Nếu là URL (random pair) → fetch thành File
      if (!faceFile && typeof faceImage === 'string') {
        const res = await fetch(faceImage);
        const blob = await res.blob();
        faceFile = new File([blob], 'face.png', { type: blob.type });
      }
      if (!hairFile && typeof hairImage === 'string') {
        const res = await fetch(hairImage);
        const blob = await res.blob();
        hairFile = new File([blob], 'hair.png', { type: blob.type });
      }

      // 1. Detect faces in face image
      setPipelineStatus('Đang quét khuôn mặt...');
      const faceResult = await detectAndSelect(faceFile, 'face');

      if (faceResult.needSelect) {
        // Tạm dừng — hiện popup chọn face
        setLoading(false);
        setPipelineStatus('');
        setFaceSelectData({
          faces: faceResult.faces,
          title: 'Chọn khuôn mặt từ ảnh chân dung',
          source: 'face',
          pendingHairFile: hairFile,
          pendingFaceFile: faceFile,
        });
        return;
      }

      // 2. Detect faces in hair image
      setPipelineStatus('Đang quét ảnh tóc...');
      const hairResult = await detectAndSelect(hairFile, 'hair');

      if (hairResult.needSelect) {
        setLoading(false);
        setPipelineStatus('');
        setFaceSelectData({
          faces: hairResult.faces,
          title: 'Chọn kiểu tóc tham khảo',
          source: 'hair',
          pendingHairFile: hairFile,
          pendingFaceFile: faceFile,
          resolvedFaceUrl: null, // dùng file gốc
        });
        return;
      }

      // 3. Generate
      await runGenerate(faceFile, hairFile);
    } catch (err) {
      setPipelineError(err.message);
      setPipelineStatus('');
      setLoading(false);
    }
  };

  // ========== Face Selected from Modal ==========
  const handleFaceSelected = async (face) => {
    const data = faceSelectData;
    setFaceSelectData(null);

    if (data.source === 'face') {
      // Đã chọn face → tiếp tục check hair
      const croppedUrl = face.cropped_image_url;
      setSelectedFaceUrl(croppedUrl);

      setLoading(true);
      setPipelineStatus('Đang quét ảnh tóc...');

      try {
        const hairResult = await detectAndSelect(data.pendingHairFile, 'hair');

        if (hairResult.needSelect) {
          setLoading(false);
          setPipelineStatus('');
          setFaceSelectData({
            faces: hairResult.faces,
            title: 'Chọn kiểu tóc tham khảo',
            source: 'hair',
            pendingHairFile: data.pendingHairFile,
            pendingFaceFile: data.pendingFaceFile,
            resolvedFaceUrl: croppedUrl,
          });
          return;
        }

        // Fetch face crop → File, rồi generate
        const faceRes = await fetch(croppedUrl);
        const faceBlob = await faceRes.blob();
        const faceFile = new File([faceBlob], 'face_crop.png', { type: faceBlob.type });

        await runGenerate(faceFile, data.pendingHairFile);
      } catch (err) {
        setPipelineError(err.message);
        setPipelineStatus('');
        setLoading(false);
      }
    } else {
      // source === 'hair': đã chọn tóc → generate
      const croppedUrl = face.cropped_image_url;
      setSelectedHairUrl(croppedUrl);
      setLoading(true);

      try {
        // Lấy face file
        let faceFile;
        if (data.resolvedFaceUrl) {
          const res = await fetch(data.resolvedFaceUrl);
          const blob = await res.blob();
          faceFile = new File([blob], 'face_crop.png', { type: blob.type });
        } else {
          faceFile = data.pendingFaceFile;
        }

        // Lấy hair crop
        const hairRes = await fetch(croppedUrl);
        const hairBlob = await hairRes.blob();
        const hairFile = new File([hairBlob], 'hair_crop.png', { type: hairBlob.type });

        await runGenerate(faceFile, hairFile);
      } catch (err) {
        setPipelineError(err.message);
        setPipelineStatus('');
        setLoading(false);
      }
    }
  };

  // ========== Generate API + Poll ==========
  const runGenerate = async (faceFile, hairFile) => {
    setPipelineStatus('Đang tạo kiểu tóc...');

    const color = selectedColor !== 'none' ? selectedColor : null;
    const { task_id } = await generateHair(faceFile, hairFile, prompt, color, colorIntensity);

    const { promise } = pollTask(task_id, (data) => {
      if (data.status === 'PROCESSING') {
        setPipelineStatus('Đang tạo kiểu tóc...');
      }
    });

    const result = await promise;
    setLoading(false);

    if (result && result.status === 'SUCCESS' && result.result_url) {
      setResultUrl(result.result_url);
      setPipelineStatus('');
      showToast('Tạo kiểu tóc thành công!', 'success');
    } else {
      setPipelineError(result?.error || 'Xử lý thất bại');
      setPipelineStatus('');
    }
  };

  // ========== Random Pair ==========
  const handleRandomPair = async () => {
    try {
      const data = await getRandomPair();
      if (data.target_url) setFaceImage(data.target_url);
      if (data.hair_url) setHairImage(data.hair_url);
      setResultUrl(null);
      setPipelineError('');
    } catch {
      showToast('Không tải được ảnh mẫu', 'error');
    }
  };

  return (
    <div className="max-w-[1200px] mx-auto px-4 py-4">
      <Header />

      {/* Main row: 2 uploads + Draw button + Result */}
      <div className="grid grid-cols-[1fr_1fr_auto_1fr] gap-4 mb-5 items-stretch">
        <ImageUpload
          label="Ảnh đầu vào 1: Chân dung"
          image={faceImage}
          onImageSelect={setFaceImage}
        />
        <ImageUpload
          label="Ảnh đầu vào 2: Tham khảo kiểu tóc"
          image={hairImage}
          onImageSelect={setHairImage}
        />
        <div className="min-w-[120px] flex items-center">
          <DrawButton onClick={handleDraw} loading={loading} />
        </div>
        <ResultPanel
          resultUrl={resultUrl}
          status={loading ? 'PROCESSING' : ''}
          error={pipelineError}
        />
      </div>

      {/* Prompt + Color row */}
      <div className="grid grid-cols-[1fr_1fr] gap-4 mb-4">
        <div className="flex flex-col gap-4">
          <PromptInput value={prompt} onChange={setPrompt} />
          <div className="flex gap-3">
            <button
              onClick={handleRandomPair}
              className="px-4 py-2 bg-gray-100 text-gray-600 rounded-lg text-sm 
                         hover:bg-gray-200 transition"
            >
              Ảnh FFHQ ngẫu nhiên
            </button>
            {pipelineStatus && (
              <span className="flex items-center text-sm text-gray-500">{pipelineStatus}</span>
            )}
          </div>
        </div>
        <ColorPicker
          selectedColor={selectedColor}
          onColorChange={setSelectedColor}
          intensity={colorIntensity}
          onIntensityChange={setColorIntensity}
        />
      </div>

      {/* Face Selector Modal */}
      {faceSelectData && (
        <FaceSelector
          faces={faceSelectData.faces}
          title={faceSelectData.title}
          onConfirm={handleFaceSelected}
          onCancel={() => { setFaceSelectData(null); setLoading(false); setPipelineStatus(''); }}
        />
      )}

      {/* Toast */}
      {toast && (
        <div className={`toast ${toast.type}`} onClick={() => setToast(null)}>
          {toast.message}
        </div>
      )}
    </div>
  );
}
