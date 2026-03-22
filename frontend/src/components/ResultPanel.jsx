export default function ResultPanel({ resultUrl, status, error }) {
  return (
    <div className="result-panel flex flex-col gap-3">
      <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Ảnh kết quả</h3>

      {status === 'PROCESSING' && (
        <div className="flex flex-col items-center gap-3 py-8">
          <div className="progress-bar w-full max-w-[200px]">
            <div className="progress-bar-fill" />
          </div>
          <p className="text-sm text-gray-500">Đang xử lý, vui lòng chờ...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-50 text-red-700 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      {resultUrl && status !== 'PROCESSING' && (
        <>
          <img src={resultUrl} alt="Kết quả" />
          <a
            href={resultUrl}
            download="hair_result.png"
            className="inline-flex items-center justify-center gap-2 bg-[#2d9b8e] text-white 
                       rounded-lg px-4 py-2.5 text-sm font-medium hover:bg-[#1a7a6d] transition"
          >
            Tải xuống kết quả
          </a>
        </>
      )}

      {!resultUrl && status !== 'PROCESSING' && !error && (
        <div className="flex items-center justify-center py-16 text-gray-300 text-sm">
          Kết quả sẽ hiển thị ở đây
        </div>
      )}
    </div>
  );
}
