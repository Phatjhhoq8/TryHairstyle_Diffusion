import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import DrawButton from './components/DrawButton';
import ResultPanel from './components/ResultPanel';
import PromptInput from './components/PromptInput';
import PromptBuilder from './components/PromptBuilder';
import PromptPreview from './components/PromptPreview';
import PromptPrioritySlider from './components/PromptPrioritySlider';
import ColorPicker from './components/ColorPicker';
import FaceSelector from './components/FaceSelector';
import ModelSelector from './components/ModelSelector';
import { generateHair, detectFaces, pollTask, getRandomPair, colorizeHair, translatePrompt } from './api/hairApi';
import { DEFAULT_PROMPT_BUILDER, buildStructuredPrompt, mergePromptParts, getModelPromptSupport } from './utils/promptUtils';

export default function App() {
  const [faceImage, setFaceImage] = useState(null);
  const [hairImage, setHairImage] = useState(null);

  const [prompt, setPrompt] = useState('');
  const [language, setLanguage] = useState('en');
  const [promptBuilder, setPromptBuilder] = useState(DEFAULT_PROMPT_BUILDER);
  const [promptPriority, setPromptPriority] = useState(50);
  const [promptPreview, setPromptPreview] = useState('high quality realistic hairstyle');
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const promptPriorityMemory = useRef(50);

  const [selectedColor, setSelectedColor] = useState('none');
  const [colorIntensity, setColorIntensity] = useState(0.7);
  const [aiModel, setAiModel] = useState('TryHairstyle');

  const [loading, setLoading] = useState(false);
  const [colorLoading, setColorLoading] = useState(false);
  const [resultUrl, setResultUrl] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState('');
  const [pipelineError, setPipelineError] = useState('');

  const [faceSelectData, setFaceSelectData] = useState(null);
  const [, setSelectedFaceUrl] = useState(null);
  const [, setSelectedHairUrl] = useState(null);

  const [toast, setToast] = useState(null);
  const toastTimer = useRef(null);

  const modelSupport = useMemo(() => getModelPromptSupport(aiModel), [aiModel]);
  const structuredPrompt = useMemo(() => buildStructuredPrompt(promptBuilder), [promptBuilder]);
  const sourcePrompt = useMemo(() => {
    const merged = mergePromptParts(structuredPrompt, prompt);
    return merged || 'high quality realistic hairstyle';
  }, [structuredPrompt, prompt]);

  useEffect(() => {
    if (aiModel === 'TryOnHairstyle') {
      if (promptPriority !== 0) {
        promptPriorityMemory.current = promptPriority;
        setPromptPriority(0);
      }
      return;
    }

    if (promptPriority === 0) {
      setPromptPriority(promptPriorityMemory.current || 50);
    }
  }, [aiModel, promptPriority]);

  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setPreviewLoading(true);
      setPreviewError('');

      try {
        if (language === 'en') {
          setPromptPreview(sourcePrompt);
        } else {
          const data = await translatePrompt(sourcePrompt, language);
          if (!controller.signal.aborted) {
            setPromptPreview(data.translated_prompt || sourcePrompt);
          }
        }
      } catch {
        if (!controller.signal.aborted) {
          setPromptPreview(sourcePrompt);
          setPreviewError('Không thể cập nhật bản dịch, sẽ dùng prompt gốc nếu cần.');
        }
      } finally {
        if (!controller.signal.aborted) {
          setPreviewLoading(false);
        }
      }
    }, 250);

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [sourcePrompt, language]);

  const showToast = useCallback((message, type = 'error') => {
    setToast({ message, type });
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 4000);
  }, []);

  const detectAndSelect = async (imageFile, source) => {
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
          resolve({ needSelect: true, faces });
        } else {
          resolve({ needSelect: false, faces });
        }
      });
    });
  };

  const handleDraw = async () => {
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
      let faceFile = faceImage instanceof File ? faceImage : null;
      let hairFile = hairImage instanceof File ? hairImage : null;

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

      setPipelineStatus('Đang quét khuôn mặt...');
      const faceResult = await detectAndSelect(faceFile, 'face');

      if (faceResult.needSelect) {
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
          resolvedFaceUrl: null,
        });
        return;
      }

      let bbox = null;
      let faceCropFile = faceFile;

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

  const handleFaceSelected = async (face) => {
    const data = faceSelectData;
    setFaceSelectData(null);

    if (data.source === 'face') {
      const croppedUrl = face.cropped_image_url;
      const bbox = face.bbox;
      setSelectedFaceUrl(croppedUrl);
      setFaceImage(croppedUrl);

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
      const croppedUrl = face.cropped_image_url;
      setSelectedHairUrl(croppedUrl);
      setHairImage(croppedUrl);
      setLoading(true);

      try {
        let faceCropFile;
        let bbox = data.resolvedFaceBbox || null;
        if (data.resolvedFaceUrl) {
          const res = await fetch(data.resolvedFaceUrl);
          const blob = await res.blob();
          faceCropFile = new File([blob], 'face_crop.png', { type: blob.type });
        } else {
          faceCropFile = data.pendingFaceFile;
        }

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

  const runGenerate = async (originalFaceFile, faceCropFile, hairFile, bbox) => {
    setPipelineStatus('Đang tạo kiểu tóc...');

    const color = selectedColor !== 'none' ? selectedColor : null;
    const promptToSend = modelSupport.promptEnabled ? promptPreview : sourcePrompt;
    const { task_id } = await generateHair(
      originalFaceFile,
      faceCropFile,
      hairFile,
      promptToSend,
      color,
      colorIntensity,
      language,
      aiModel,
      bbox,
      promptPriority,
    );

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

  const handleQuickColorize = async () => {
    const sourceUrl = resultUrl;
    const sourceFile = faceImage;

    if (!sourceUrl && !sourceFile) {
      showToast('Chưa có ảnh nào để đổi màu! Hãy tải ảnh chân dung lên.');
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
      let imageFile;

      if (sourceUrl) {
        const res = await fetch(sourceUrl);
        const blob = await res.blob();
        imageFile = new File([blob], 'result.png', { type: blob.type });
      } else if (sourceFile instanceof File) {
        imageFile = sourceFile;
      } else if (typeof sourceFile === 'string') {
        const res = await fetch(sourceFile);
        const blob = await res.blob();
        imageFile = new File([blob], 'face.png', { type: blob.type });
      }

      const { task_id } = await colorizeHair(imageFile, selectedColor, colorIntensity);
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
    <div className="mx-auto max-w-[1200px] px-4 py-4">
      <Header />

      <div className="mb-5 grid items-stretch gap-4 xl:grid-cols-[1fr_1fr_auto_1fr]">
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
        <div className="flex min-w-[120px] items-center">
          <DrawButton
            onClick={handleDraw}
            loading={loading}
            onQuickColor={handleQuickColorize}
            colorLoading={colorLoading}
            hasResult={!!resultUrl || !!faceImage}
          />
        </div>
        <ResultPanel
          resultUrl={resultUrl}
          onClear={() => setResultUrl(null)}
          status={loading ? 'PROCESSING' : ''}
          error={pipelineError}
        />
      </div>

      <div className="mb-4 grid gap-4 xl:grid-cols-[1.3fr_1fr]">
        <div className="space-y-4">
          <ModelSelector value={aiModel} onChange={setAiModel} />
          <PromptBuilder value={promptBuilder} onChange={setPromptBuilder} disabled={!modelSupport.promptEnabled} />
          <PromptInput value={prompt} onChange={setPrompt} language={language} onLanguageChange={setLanguage} disabled={!modelSupport.promptEnabled} />
          <PromptPreview
            sourcePrompt={sourcePrompt}
            translatedPrompt={promptPreview}
            isLoading={previewLoading}
            translationNote={previewError}
            modelSupportsPrompt={modelSupport.promptEnabled}
          />
        </div>

        <div className="space-y-4">
          <PromptPrioritySlider
            value={modelSupport.promptEnabled ? promptPriority : 0}
            onChange={setPromptPriority}
            disabled={!modelSupport.promptEnabled}
          />
          <ColorPicker
            selectedColor={selectedColor}
            onColorChange={setSelectedColor}
            intensity={colorIntensity}
            onIntensityChange={setColorIntensity}
          />

          <div className="rounded-2xl border border-gray-200 bg-white p-4">
            <div className="flex flex-wrap gap-3">
              <button
                onClick={handleRandomPair}
                className="rounded-lg bg-gray-100 px-4 py-2 text-sm text-gray-600 transition hover:bg-gray-200"
              >
                Ảnh FFHQ ngẫu nhiên
              </button>
              <button
                onClick={() => {
                  setPrompt('');
                  setPromptBuilder({ ...DEFAULT_PROMPT_BUILDER });
                  setPromptPriority(aiModel === 'TryOnHairstyle' ? 0 : 50);
                }}
                className="rounded-lg bg-gray-100 px-4 py-2 text-sm text-gray-600 transition hover:bg-gray-200"
              >
                Đặt lại prompt
              </button>
            </div>

            {pipelineStatus && (
              <div className="mt-3 rounded-xl bg-[#f5fbfa] px-3 py-2 text-sm text-gray-600">
                {pipelineStatus}
              </div>
            )}
          </div>
        </div>
      </div>

      {faceSelectData && (
        <FaceSelector
          faces={faceSelectData.faces}
          title={faceSelectData.title}
          onConfirm={handleFaceSelected}
          onCancel={() => { setFaceSelectData(null); setLoading(false); setPipelineStatus(''); }}
        />
      )}

      {toast && (
        <div className={`toast ${toast.type}`} onClick={() => setToast(null)}>
          {toast.message}
        </div>
      )}
    </div>
  );
}
