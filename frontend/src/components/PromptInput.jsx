export default function PromptInput({ value, onChange, language, onLanguageChange }) {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <label className="text-sm font-semibold text-gray-700">Text Prompt</label>
        <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-0.5">
          <button
            onClick={() => onLanguageChange('en')}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-all duration-200
              ${language === 'en'
                ? 'bg-white text-[#2d9b8e] shadow-sm'
                : 'text-gray-500 hover:text-gray-700'}`}
          >
            🇬🇧 English
          </button>
          <button
            onClick={() => onLanguageChange('vi')}
            className={`px-3 py-1 rounded-md text-xs font-medium transition-all duration-200
              ${language === 'vi'
                ? 'bg-white text-[#2d9b8e] shadow-sm'
                : 'text-gray-500 hover:text-gray-700'}`}
          >
            🇻🇳 Tiếng Việt
          </button>
        </div>
      </div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={
          language === 'vi'
            ? 'Mô tả kiểu tóc bằng tiếng Việt (sẽ tự động dịch sang tiếng Anh)...'
            : 'Describe your desired hairstyle in English...'
        }
        rows={2}
        className="w-full rounded-lg border border-gray-200 px-4 py-3 text-sm resize-none
                   focus:outline-none focus:ring-2 focus:ring-[#2d9b8e]/30 focus:border-[#2d9b8e]
                   transition placeholder:text-gray-300"
      />
      {language === 'vi' && (
        <p className="text-xs text-gray-400 flex items-center gap-1">
          <span>🌐</span> Prompt sẽ được tự động dịch sang tiếng Anh trước khi xử lý
        </p>
      )}
    </div>
  );
}
