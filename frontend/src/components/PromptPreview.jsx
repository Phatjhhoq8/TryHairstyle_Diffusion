export default function PromptPreview({ sourcePrompt, translatedPrompt, isLoading, translationNote, modelSupportsPrompt }) {
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <div className="rounded-2xl border border-gray-200 bg-white p-4">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-gray-800">Prompt hiện tại</h3>
          <span className="rounded-full bg-gray-100 px-2.5 py-1 text-[11px] font-medium uppercase tracking-wide text-gray-500">
            Input
          </span>
        </div>
        <p className="min-h-14 text-sm leading-6 text-gray-700">
          {sourcePrompt || 'Chưa có prompt. Bạn có thể chọn tag ở Prompt Builder hoặc nhập prompt tự do.'}
        </p>
      </div>

      <div className="rounded-2xl border border-[#cfe6e1] bg-[#f5fbfa] p-4">
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-gray-800">Prompt gửi vào model</h3>
          <span className="rounded-full bg-[#e6f6f3] px-2.5 py-1 text-[11px] font-medium uppercase tracking-wide text-[#1a7a6d]">
            English
          </span>
        </div>
        <p className="min-h-14 text-sm leading-6 text-gray-700">
          {isLoading ? 'Đang cập nhật bản dịch...' : (translatedPrompt || sourcePrompt || 'Không có dữ liệu để preview.')}
        </p>
        <p className="mt-2 text-xs text-gray-500">
          {!modelSupportsPrompt
            ? 'TryOnHairstyle hiện khóa prompt priority ở 0% và bám ảnh tham chiếu là chính.'
            : (translationNote || 'Bản này giúp kiểm tra prompt cuối trước khi generate.')}
        </p>
      </div>
    </div>
  );
}
