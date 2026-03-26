import { useState, useCallback, useRef } from 'react';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import DrawButton from './components/DrawButton';
import ResultPanel from './components/ResultPanel';
import PromptInput from './components/PromptInput';
import ColorPicker from './components/ColorPicker';
import FaceSelector from './components/FaceSelector';
import ModelSelector from './components/ModelSelector';
import { generateHair, detectFaces, pollTask, getRandomPair, colorizeHair } from './api/hairApi';

export default function App() {
  // Images (File objects hoặc URL string)
  const [faceImage, setFaceImage] = useState(null);
  const [hairImage, setHairImage] = useState(null);

  // Prompt + Color
  const [prompt, setPrompt] = useState('high quality, realistic hairstyle');
  const [language, setLanguage] = useState('en');
  const [selectedColor, setSelectedColor] = useState('none');
  const [colorIntensity, setColorIntensity] = useState(0.7);
  const [aiModel, setAiModel] = useState('HairFusion');

  // Pipeline state
  const [loading, setLoading] = useState(false);
  const [colorLoading, setColorLoading] = useState(false);
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
      // Mặc định nếu không có bbox (không tìm thấy mặt), truyền null
      let bbox = null;
      let faceCropFile = faceFile;
      
      // Nếu tìm thấy chính xác 1 mặt:
      if (faceResult.faces && faceResult.faces.length === 1) {
        bbox = faceResult.faces[0].bbox;
        const faceRes = await fetch(faceResult.faces[0].cropped_image_url);
        const faceBlob = await faceRes.blob();
        faceCropFile = new File([faceBlob], 'face_crop.png', { type: faceBlob.type });
      }

      await runGenerate(faceFile, faceCropFile, hairFile, bbox);
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
      const bbox = face.bbox;
      setSelectedFaceUrl(croppedUrl);
      setFaceImage(croppedUrl);  // Cập nhật ảnh ở ô input

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
            resolvedFaceBbox: bbox,
          });
          return;
        }

        // Fetch face crop → File, rồi generate
        const faceRes = await fetch(croppedUrl);
        const faceBlob = await faceRes.blob();
        const faceFile = new File([faceBlob], 'face_crop.png', { type: faceBlob.type });

        await runGenerate(data.pendingFaceFile, faceFile, data.pendingHairFile, bbox);
      } catch (err) {
        setPipelineError(err.message);
        setPipelineStatus('');
        setLoading(false);
      }
    } else {
      // source === 'hair': đã chọn tóc → generate
      const croppedUrl = face.cropped_image_url;
      setSelectedHairUrl(croppedUrl);
      setHairImage(croppedUrl);  // Cập nhật ảnh ở ô input
      setLoading(true);

      try {
        // Lấy face file
        let faceFile;
        let faceCropFile;
        let bbox = data.resolvedFaceBbox || null;
        if (data.resolvedFaceUrl) {
          const res = await fetch(data.resolvedFaceUrl);
          const blob = await res.blob();
          faceCropFile = new File([blob], 'face_crop.png', { type: blob.type });
        } else {
          // If 1 face was detected originally, use it
          if (data.faces && data.faces.length === 1 && data.source !== 'hair') {
            bbox = data.faces[0].bbox;
            const res = await fetch(data.faces[0].cropped_image_url);
            const blob = await res.blob();
            faceCropFile = new File([blob], 'face_crop.png', { type: blob.type });
          } else {
            faceCropFile = data.pendingFaceFile;
          }
        }

        // Lấy hair crop
        const hairRes = await fetch(croppedUrl);
        const hairBlob = await hairRes.blob();
        const hairFile = new File([hairBlob], 'hair_crop.png', { type: hairBlob.type });

        await runGenerate(data.pendingFaceFile, faceCropFile, hairFile, bbox);
      } catch (err) {
        setPipelineError(err.message);
        setPipelineStatus('');
        setLoading(false);
      }
    }
  };

  // ========== Generate API + Poll ==========
  const runGenerate = async (originalFaceFile, faceCropFile, hairFile, bbox) => {
    setPipelineStatus('Đang tạo kiểu tóc...');

    const color = selectedColor !== 'none' ? selectedColor : null;
    const { task_id } = await generateHair(originalFaceFile, faceCropFile, hairFile, prompt, color, colorIntensity, language, aiModel, bbox);

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

  // ========== Quick Colorize (Đổi màu nhanh) ==========
  const handleQuickColorize = async () => {
    if (!resultUrl) {
      showToast('Chưa có ảnh kết quả để đổi màu!');
      return;
    }
    if (!selectedColor || selectedColor === 'none') {
      showToast('Vui lòng chọn màu tóc trước khi đổi màu!', 'info');
      return;
    }

    setColorLoading(true);
    setPipelineError('');
    setPipelineStatus('Đang đổi màu tóc...');

    try {
      // Fetch ảnh kết quả hiện tại → File
      const res = await fetch(resultUrl);
      const blob = await res.blob();
      const imageFile = new File([blob], 'result.png', { type: blob.type });

      // Gọi API /colorize
      const { task_id } = await colorizeHair(imageFile, selectedColor, colorIntensity);

      // Poll task
      const { promise } = pollTask(task_id, (data) => {
        if (data.status === 'PROCESSING') {
          setPipelineStatus('Đang đổi màu tóc...');
        }
      });

      const result = await promise;
      setColorLoading(false);

      if (result && result.status === 'SUCCESS' && result.result_url) {
        setResultUrl(result.result_url);
        setPipelineStatus('');
        showToast('Đổi màu tóc thành công!', 'success');
      } else {
        setPipelineError(result?.error || 'Đổi màu thất bại');
        setPipelineStatus('');
      }
    } catch (err) {
      setPipelineError(err.message);
      setPipelineStatus('');
      setColorLoading(false);
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
          <DrawButton
            onClick={handleDraw}
            loading={loading}
            onQuickColor={handleQuickColorize}
            colorLoading={colorLoading}
            hasResult={!!resultUrl}
          />
        </div>
        <ResultPanel
          resultUrl={resultUrl}
          status={loading ? 'PROCESSING' : ''}
          error={pipelineError}
        />
      </div>

      {/* Prompt row */}
      <div className="mb-4">
        <PromptInput value={prompt} onChange={setPrompt} language={language} onLanguageChange={setLanguage} />
        <div className="flex gap-3 mt-2">
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

      {/* Model Selector + Color Picker row */}
      <div className="grid grid-cols-[1fr_1fr] gap-4 mb-4">
        <ModelSelector value={aiModel} onChange={setAiModel} />
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
