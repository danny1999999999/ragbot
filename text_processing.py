from core_config import *
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict

# LangChain相關
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# EPUB處理相關 (條件導入)
if EPUB_AVAILABLE:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

# 中文處理相關 (條件導入)
if JIEBA_AVAILABLE:
    import jieba
if OPENCC_AVAILABLE:
    import opencc

# 配置日誌
logger = logging.getLogger(__name__)

class AdvancedTokenEstimator:
    """🔧 高級 Token 估算器"""
    
    def __init__(self):
        # 不同語言和內容類型的 Token 係數
        self.token_ratios = {
            "chinese": 2.5,      # 中文字符/token
            "english": 4.0,      # 英文字符/token
            "mixed": 3.0,        # 中英混合
            "code": 3.5,         # 程式碼
            "punctuation": 1.0,  # 標點符號
            "numbers": 2.0       # 數字
        }
        self.safety_margin = TOKEN_LIMITS.get("token_safety_margin", 0.15)
    
    def analyze_text_composition(self, text: str) -> Dict[str, int]:
        """分析文本組成"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        numbers = len(re.findall(r'\d', text))
        punctuation = len(re.findall(r'[^\w\s\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars - english_chars - numbers - punctuation
        
        return {
            "chinese": chinese_chars,
            "english": english_chars,
            "numbers": numbers,
            "punctuation": punctuation,
            "other": other_chars,
            "total": len(text)
        }
    
    def estimate_tokens(self, text: str, content_type: str = "mixed") -> int:
        """精確估算 Token 數量"""
        if not text:
            return 0
        
        composition = self.analyze_text_composition(text)
        
        # 根據文本組成計算 token
        estimated_tokens = (
            composition["chinese"] / self.token_ratios["chinese"] +
            composition["english"] / self.token_ratios["english"] +
            composition["numbers"] / self.token_ratios["numbers"] +
            composition["punctuation"] / self.token_ratios["punctuation"] +
            composition["other"] / self.token_ratios["mixed"]
        )
        
        # 根據內容類型調整
        if content_type == "code":
            estimated_tokens *= 1.2  # 程式碼通常 token 密度更高
        elif content_type == "academic":
            estimated_tokens *= 1.1  # 學術文本專業詞彙多
        
        # 加上安全邊際
        final_estimate = int(estimated_tokens * (1 + self.safety_margin))
        
        return max(final_estimate, 1)  # 至少1個token
    
    def estimate_embedding_cost(self, total_tokens: int, model: str = "text-embedding-3-small") -> float:
        """估算 Embedding 成本"""
        cost_per_1k_tokens = {
            "text-embedding-3-small": 0.00002,
            "text-embedding-3-large": 0.00013,
            "text-embedding-ada-002": 0.0001
        }
        
        rate = cost_per_1k_tokens.get(model, 0.00002)
        return (total_tokens / 1000) * rate

class ChineseTextNormalizer:
    """優化版中文文本標準化處理器"""
    
    def __init__(self):
        self.s2t_converter = None
        self.t2s_converter = None
        
        if OPENCC_AVAILABLE:
            try:
                self.s2t_converter = opencc.OpenCC('s2t')
                self.t2s_converter = opencc.OpenCC('t2s')
                print("🔤 繁簡轉換器初始化完成")
            except Exception as e:
                logger.warning(f"OpenCC 初始化失敗: {e}")
        
        # 中文文本正規化規則
        self.normalization_rules = [
            (r'[\u3000]+', ' '),
            (r'\r\n|\r', '\n'),                       # 統一換行
            (r'\n{3,}', '\n\n'),                      # 限制連續換行
            (r'[\u201C\u201D\u2018\u2019\u201E\u201A\u2033\u2032]', '"'),  # 統一引號
            (r'[——–∶]', '-'),                          # 統一破折號
            (r'[…⋯]', '...'),                         # 統一省略號
        ]
    
    def normalize_text(self, text: str) -> Tuple[str, Dict]:
        """標準化文本並返回處理資訊"""
        if not text:
            return "", {}
        
        original_length = len(text)
        processed_text = text
        
        # 應用正規化規則
        for pattern, replacement in self.normalization_rules:
            processed_text = re.sub(pattern, replacement, processed_text)
        
        # 移除首尾空白
        processed_text = processed_text.strip()
        
        # 檢測和轉換繁簡
        variant, confidence = self.detect_chinese_variant(processed_text)
        if variant == 'simplified' and self.s2t_converter:
            try:
                processed_text = self.s2t_converter.convert(processed_text)
                variant = 'traditional_converted'
            except Exception as e:
                logger.warning(f"繁簡轉換失敗: {e}")
        
        processing_info = {
            'original_length': original_length,
            'processed_length': len(processed_text),
            'variant': variant,
            'confidence': confidence,
            'normalized': original_length != len(processed_text)
        }
        
        return processed_text, processing_info
    
    def detect_chinese_variant(self, text: str) -> Tuple[str, float]:
        """檢測繁體或簡體中文"""
        if not text:
            return 'unknown', 0.0
        
        simplified_chars = set('国发会学习论问题业专长时间经济')
        traditional_chars = set('國發會學習論問題業專長時間經濟')
        
        simplified_count = sum(1 for char in text if char in simplified_chars)
        traditional_count = sum(1 for char in text if char in traditional_chars)
        total_chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
        
        if total_chinese == 0:
            return 'unknown', 0.0
        
        simplified_ratio = simplified_count / total_chinese
        traditional_ratio = traditional_count / total_chinese
        
        if simplified_ratio > traditional_ratio:
            return 'simplified', simplified_ratio
        elif traditional_ratio > simplified_ratio:
            return 'traditional', traditional_ratio
        else:
            return 'mixed', max(simplified_ratio, traditional_ratio)
    
    def create_search_variants(self, query: str) -> List[str]:
        """創建繁簡搜索變體"""
        variants = [query]
        
        if not self.s2t_converter or not self.t2s_converter:
            return variants
        
        try:
            traditional = self.s2t_converter.convert(query)
            if traditional != query:
                variants.append(traditional)
            
            simplified = self.t2s_converter.convert(query)
            if simplified != query:
                variants.append(simplified)
        except Exception as e:
            logger.warning(f"創建搜索變體失敗: {e}")
        
        return list(set(variants))

class EpubProcessor:
    """📚 EPUB 文件處理器"""
    
    def __init__(self):
        self.normalizer = ChineseTextNormalizer()
        
    def extract_epub_content(self, file_path: Path) -> str:
        """提取 EPUB 內容"""
        try:
            if not EPUB_AVAILABLE:
                raise ImportError("EPUB 處理庫未安裝")
            
            print(f"📚 正在處理 EPUB 文件: {file_path.name}")
            
            # 讀取 EPUB 文件
            book = epub.read_epub(str(file_path))
            
            # 提取所有文本內容
            content_parts = []
            chapter_count = 0
            
            # 獲取書籍信息
            title = book.get_metadata('DC', 'title')
            author = book.get_metadata('DC', 'creator')
            
            book_info = []
            if title:
                book_info.append(f"書名: {title[0][0] if title else '未知'}")
            if author:
                book_info.append(f"作者: {author[0][0] if author else '未知'}")
            
            if book_info:
                content_parts.append("\n".join(book_info) + "\n\n")
            
            # 按順序處理章節
            spine_items = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
            
            for item in spine_items:
                try:
                    # 解析 HTML 內容
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    
                    # 移除腳本和樣式
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # 提取文本
                    text = soup.get_text()
                    
                    if text and len(text.strip()) > 50:  # 過濾太短的內容
                        # 清理文本
                        cleaned_text = self._clean_epub_text(text)
                        
                        if cleaned_text.strip():
                            chapter_count += 1
                            
                            # 添加章節標記
                            chapter_title = self._extract_chapter_title(soup, cleaned_text)
                            if chapter_title:
                                content_parts.append(f"\n\n=== {chapter_title} ===\n")
                            else:
                                content_parts.append(f"\n\n=== 第 {chapter_count} 章 ===\n")
                            
                            content_parts.append(cleaned_text)
                            
                except Exception as e:
                    print(f"   ⚠️ 章節處理失敗: {e}")
                    continue
            
            if not content_parts:
                raise ValueError("EPUB 文件中沒有找到有效內容")
            
            full_content = "".join(content_parts)
            
            print(f"   ✅ EPUB 處理完成: {chapter_count} 個章節, {len(full_content):,} 字符")
            
            return full_content
            
        except Exception as e:
            print(f"   ❌ EPUB 處理失敗: {e}")
            raise
    
    def _clean_epub_text(self, text: str) -> str:
        """清理 EPUB 文本"""
        if not text:
            return ""
        
        # 標準化文本
        normalized_text, _ = self.normalizer.normalize_text(text)
        
        # EPUB 特定清理
        epub_cleaning_rules = [
            (r'\n{4,}', '\n\n'),                    # 限制連續換行
            (r'[ \t]{3,}', ' ' ),                    # 限制連續空格
            (r'^[ \t]+', '', re.MULTILINE),          # 移除行首空白
            (r'[ \t]+$', '', re.MULTILINE),          # 移除行尾空白
            (r'\n[ \t]*\n', '\n\n'),                # 清理空行
            (r'[\x00-\x08\x0B\x0C\x0E-\x1F]', ''),  # 移除控制字符
        ]
        
        cleaned_text = normalized_text
        for pattern, replacement, *flags in epub_cleaning_rules:
            flag = flags[0] if flags else 0
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=flag)
        
        return cleaned_text.strip()
    
    def _extract_chapter_title(self, soup, text: str) -> Optional[str]:
        """嘗試提取章節標題"""
        try:
            # 從 HTML 標籤中查找標題
            for tag in ['h1', 'h2', 'h3', 'title']:
                title_element = soup.find(tag)
                if title_element and title_element.get_text().strip():
                    title = title_element.get_text().strip()
                    if 5 <= len(title) <= 100:  # 合理的標題長度
                        return title
            
            # 從文本開頭查找標題
            lines = text.split('\n')[:5]  # 檢查前5行
            for line in lines:
                line = line.strip()
                if line and 5 <= len(line) <= 100:
                    # 檢查是否包含章節關鍵詞
                    chapter_keywords = ['章', 'Chapter', '第', '卷', 'Part']
                    if any(keyword in line for keyword in chapter_keywords):
                        return line
                    # 或者是短行且看起來像標題
                    elif len(line) <= 50 and not line.endswith(('。', '！', '？', '.', '!', '?')):
                        return line
            
            return None
            
        except Exception:
            return None

class SmartTextAnalyzer:
    """🧠 智能文本分析器"""
    
    def __init__(self):
        self.normalizer = ChineseTextNormalizer()
        self.token_estimator = AdvancedTokenEstimator()
        
    def analyze_text(self, text: str, source_info: Dict = None) -> TextAnalysis:
        """綜合分析文本"""
        if not text:
            return TextAnalysis(
                length=0, text_type="empty", language="unknown",
                encoding="utf-8", structure_info={}, quality_score=0.0,
                processing_strategy="skip"
            )
        
        # 標準化文本
        normalized_text, norm_info = self.normalizer.normalize_text(text)
        length = len(normalized_text)
        
        # 分類文本長度
        text_type = self._classify_text_length(length)
        
        # 分析文本結構
        structure_info = self._analyze_structure(normalized_text)
        
        # 檢測語言
        language = self._detect_language(normalized_text)
        
        # 評估質量
        quality_score = self._evaluate_quality(normalized_text, structure_info)
        
        # 決定處理策略
        processing_strategy = self._determine_strategy(text_type, structure_info, quality_score)
        
        return TextAnalysis(
            length=length,
            text_type=text_type,
            language=language,
            encoding=norm_info.get('encoding', 'utf-8'),
            structure_info=structure_info,
            quality_score=quality_score,
            processing_strategy=processing_strategy
        )
    
    def _classify_text_length(self, length: int) -> str:
        """分類文本長度"""
        config = SMART_TEXT_CONFIG
        
        if length < config["micro_text_threshold"]:
            return "micro_text"
        elif length < config["short_text_threshold"]:
            return "short_text"
        elif length < config["medium_text_threshold"]:
            return "medium_text"
        elif length < config["long_text_threshold"]:
            return "long_text"
        elif length < config["ultra_long_threshold"]:
            return "ultra_long"
        else:
            return "mega_text"  # 超大文本
    
    def _analyze_structure(self, text: str) -> Dict:
        """分析文本結構"""
        structure_info = {
            'paragraphs': len(text.split('\n\n')),
            'lines': len(text.split('\n')),
            'sentences': len(re.findall(r'[。！？.!?]+', text)),
            'has_chapters': False,
            'has_sections': False,
            'has_lists': False,
            'has_tables': False,
            'chapter_count': 0,
            'section_count': 0
        }
        
        # 檢測章節結構
        chapter_patterns = [
            r'第[一二三四五六七八九十\d]+章',
            r'Chapter\s+\d+',
            r'\d+\.\s*[^\n]{1,50}\n'
        ]
        
        for pattern in chapter_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                structure_info['has_chapters'] = True
                structure_info['chapter_count'] = len(matches)
                break
        
        # 檢測小節結構
        section_patterns = [
            r'第[一二三四五六七八九十\d]+節',
            r'\d+\.\d+',
            r'[一二三四五六七八九十]、'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, text)
            if matches:
                structure_info['has_sections'] = True
                structure_info['section_count'] = len(matches)
                break
        
        # 檢測列表
        if re.search(r'^\s*[•\-*]\s+', text, re.MULTILINE) or \
           re.search(r'^\s*\d+\.\s+', text, re.MULTILINE):
            structure_info['has_lists'] = True
        
        # 檢測表格
        if '|' in text and text.count('|') > 4:
            structure_info['has_tables'] = True
        
        return structure_info
    
    def _detect_language(self, text: str) -> str:
        """檢測文本主要語言"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = len(text)
        
        if total_chars == 0:
            return "unknown"
        
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if chinese_ratio > 0.3:
            if english_ratio > 0.2:
                return "mixed_zh_en"
            else:
                return "chinese"
        elif english_ratio > 0.5:
            return "english"
        else:
            return "mixed"
    
    def _evaluate_quality(self, text: str, structure_info: Dict) -> float:
        """評估文本質量（0-1分數）"""
        score = 0.5  # 基礎分數
        
        # 長度合理性（太短或太長都扣分）
        length = len(text)
        if 50 <= length <= 10000:
            score += 0.2
        elif length < 20:
            score -= 0.3
        
        # 結構完整性
        if structure_info['paragraphs'] > 1:
            score += 0.1
        if structure_info['sentences'] > 2:
            score += 0.1
        
        # 內容豐富度
        unique_chars = len(set(text))
        if unique_chars > 20:
            score += 0.1
        
        # 避免重複內容
        lines = text.split('\n')
        unique_lines = len(set(lines))
        if len(lines) > 0:
            uniqueness_ratio = unique_lines / len(lines)
            score += uniqueness_ratio * 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _determine_strategy(self, text_type: str, structure_info: Dict, quality_score: float) -> str:
        """決定處理策略"""
        if quality_score < 0.3:
            return "low_quality_skip"
        
        if text_type in ["micro_text", "short_text"]:
            return "whole_document"
        elif text_type == "medium_text":
            if structure_info['has_chapters'] or structure_info['paragraphs'] > 5:
                return "paragraph_aware"
            else:
                return "simple_split"
        elif text_type == "long_text":
            if structure_info['has_chapters']:
                return "chapter_aware"
            elif structure_info['has_sections']:
                return "section_aware"
            else:
                return "fine_split"
        elif text_type in ["ultra_long", "mega_text"]:
            return "hierarchical_split"
        else:
            # 回退策略
            return "hierarchical_split"

class OptimizedTextSplitter:
    """🔧 優化版文本分割器"""
    
    def __init__(self):
        self.analyzer = SmartTextAnalyzer()
        self.token_estimator = AdvancedTokenEstimator()
        self.config = SMART_TEXT_CONFIG
        
        # 分割器實例池
        self._splitter_cache = {}
        
        print("🔧 優化版文本分割器初始化完成")
        print(f"   📝 支持 {len(self.config)} 種文本類型")
        print(f"   🧠 智能策略選擇")
        print(f"   ⚡ 性能優化")
    
    def get_splitter(self, chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
        """獲取分割器實例（帶快取）"""
        cache_key = f"{chunk_size}_{chunk_overlap}"
        
        if cache_key not in self._splitter_cache:
            self._splitter_cache[cache_key] = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n第", "\n第",      # 章節標題
                    "\n\n", "\n",          # 段落
                    "。", "！", "？", "；",  # 句子結束
                    "，", "、",            # 短語分隔
                    " ", ""
                ],
                keep_separator=True,
                length_function=len
            )
        
        return self._splitter_cache[cache_key]
    
    def smart_split_documents(self, text: str, doc_id: str, source_info: Dict = None) -> List[Document]:
        """🧠 智能文檔分割主入口"""
        if not text or not text.strip():
            return []
        
        # 分析文本
        analysis = self.analyzer.analyze_text(text, source_info)
        
        print(f"📊 文本分析: {analysis.text_type} ({analysis.length:,} 字符)")
        print(f"   🎯 處理策略: {analysis.processing_strategy}")
        print(f"   📈 質量分數: {analysis.quality_score:.2f}")
        print(f"   🗣️ 主要語言: {analysis.language}")
        
        # 根據策略選擇處理方法
        documents = []
        if analysis.processing_strategy == "low_quality_skip":
            print("   ⚠️ 文本質量過低，跳過處理")
            return []
        elif analysis.processing_strategy == "whole_document":
            documents = self._process_whole_document(text, doc_id, analysis)
        elif analysis.processing_strategy == "paragraph_aware":
            documents = self._process_paragraph_aware(text, doc_id, analysis)
        elif analysis.processing_strategy == "simple_split":
            documents = self._process_simple_split(text, doc_id, analysis)
        elif analysis.processing_strategy == "chapter_aware":
            documents = self._process_chapter_aware(text, doc_id, analysis)
        elif analysis.processing_strategy == "section_aware":
            documents = self._process_section_aware(text, doc_id, analysis)
        elif analysis.processing_strategy == "fine_split":
            documents = self._process_fine_split(text, doc_id, analysis)
        elif analysis.processing_strategy == "hierarchical_split":
            documents = self._process_hierarchical_split(text, doc_id, analysis)
        else:
            # 回退到簡單分割
            documents = self._process_simple_split(text, doc_id, analysis)

        # 最終驗證和過濾，確保沒有空內容的文檔
        final_documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        if len(final_documents) != len(documents):
            logger.warning(f"過濾掉 {len(documents) - len(final_documents)} 個空內容的倫塊")
        
        return final_documents
    
    def _process_whole_document(self, text: str, doc_id: str, analysis: TextAnalysis) -> List[Document]:
        """處理整個文檔（不分割）"""
        config = self.config[analysis.text_type]
        
        if len(text.strip()) < config.get("min_length", 20):
            print(f"   ⚠️ 文檔過短，跳過處理")
            return []
        
        print(f"   📝 {config['description']}")
        
        # 計算 token 數
        token_count = self.token_estimator.estimate_tokens(text)
        
        metadata = {
            'doc_id': doc_id,
            'chunk_id': f"{doc_id}_whole",
            'chunk_index': 0,
            'total_chunks': 1,
            'text_type': analysis.text_type,
            'processing_strategy': analysis.processing_strategy,
            'is_complete_document': True,
            'token_count': token_count,
            'quality_score': analysis.quality_score,
            'language': analysis.language,
            'structure_info': analysis.structure_info
        }
        
        return [Document(page_content=text, metadata=metadata)]
    
    def _process_paragraph_aware(self, text: str, doc_id: str, analysis: TextAnalysis) -> List[Document]:
        """段落感知處理"""
        config = self.config[analysis.text_type]
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        
        print(f"   📒 {config['description']} (目標大小: {chunk_size})")
        
        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            return self._process_simple_split(text, doc_id, analysis)
        
        documents = []
        current_chunk = ""
        chunk_index = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            # 檢查是否可以加入當前分塊
            potential_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if len(potential_chunk) <= chunk_size or not current_chunk:
                current_chunk = potential_chunk
            else:
                # 當前分塊已滿，創建文檔
                if current_chunk:
                    doc = self._create_chunk_document(
                        current_chunk, doc_id, chunk_index, analysis, "paragraph_aware"
                    )
                    documents.append(doc)
                    chunk_index += 1
                
                current_chunk = paragraph
        
        # 處理最後一個分塊
        if current_chunk:
            doc = self._create_chunk_document(
                current_chunk, doc_id, chunk_index, analysis, "paragraph_aware"
            )
            documents.append(doc)
        
        # 處理重疊
        if chunk_overlap > 0 and len(documents) > 1:
            documents = self._add_overlap_to_documents(documents, chunk_overlap)
        
        return documents
    
    def _process_simple_split(self, text: str, doc_id: str, analysis: TextAnalysis) -> List[Document]:
        """簡單分割處理"""
        config = self.config[analysis.text_type]
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        
        print(f"   ✂️ {config['description']} (大小: {chunk_size}, 重疊: {chunk_overlap})")
        
        splitter = self.get_splitter(chunk_size, chunk_overlap)
        chunks = splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) >= config.get("min_length", 20):
                doc = self._create_chunk_document(
                    chunk, doc_id, i, analysis, "simple_split"
                )
                documents.append(doc)
        
        return documents
    
    def _process_chapter_aware(self, text: str, doc_id: str, analysis: TextAnalysis) -> List[Document]:
        """章節感知處理"""
        config = self.config[analysis.text_type]
        
        print(f"   📚 {config['description']} - 檢測到 {analysis.structure_info.get('chapter_count', 0)} 個章節")
        
        # 按章節分割
        chapters = self._split_by_chapters(text)
        
        if len(chapters) <= 1:
            # 沒有檢測到章節，回退到段落感知
            return self._process_paragraph_aware(text, doc_id, analysis)
        
        all_documents = []
        
        for chapter_idx, chapter_text in enumerate(chapters):
            chapter_doc_id = f"{doc_id}_ch{chapter_idx+1:02d}"
            
            # 每個章節內部使用段落感知分割
            chapter_analysis = analysis  # 使用相同的分析結果
            chapter_docs = self._process_paragraph_aware(chapter_text, chapter_doc_id, chapter_analysis)
            
            # 添加章節信息到元數據
            for doc in chapter_docs:
                doc.metadata.update({
                    'chapter_index': chapter_idx,
                    'chapter_total': len(chapters),
                    'parent_doc_id': doc_id,
                    'split_method': 'chapter_aware'
                })
            
            all_documents.extend(chapter_docs)
        
        return all_documents
    
    def _process_section_aware(self, text: str, doc_id: str, analysis: TextAnalysis) -> List[Document]:
        """小節感知處理"""
        config = self.config[analysis.text_type]
        
        print(f"   📋 {config['description']} - 檢測到 {analysis.structure_info.get('section_count', 0)} 個小節")
        
        # 按小節分割
        sections = self._split_by_sections(text)
        
        if len(sections) <= 1:
            return self._process_simple_split(text, doc_id, analysis)
        
        all_documents = []
        
        for section_idx, section_text in enumerate(sections):
            section_doc_id = f"{doc_id}_sec{section_idx+1:02d}"
            
            # 小節內部簡單分割
            section_docs = self._process_simple_split(section_text, section_doc_id, analysis)
            
            # 添加小節信息
            for doc in section_docs:
                doc.metadata.update({
                    'section_index': section_idx,
                    'section_total': len(sections),
                    'parent_doc_id': doc_id,
                    'split_method': 'section_aware'
                })
            
            all_documents.extend(section_docs)
        
        return all_documents
    
    def _process_fine_split(self, text: str, doc_id: str, analysis: TextAnalysis) -> List[Document]:
        """精細分割處理"""
        config = self.config[analysis.text_type]
        
        print(f"   🔬 {config['description']}")
        
        return self._process_simple_split(text, doc_id, analysis)
    
    def _process_hierarchical_split(self, text: str, doc_id: str, analysis: TextAnalysis) -> List[Document]:
        """階層式分割處理"""
        config = self.config[analysis.text_type]
        
        print(f"   🏗️ {config['description']}")
        
        # 第一層：章節分割
        chapters = self._split_by_chapters(text)
        
        if len(chapters) > 1:
            return self._process_chapter_aware(text, doc_id, analysis)
        
        # 第二層：段落分割
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) > 10:  # 段落較多，用段落感知
            return self._process_paragraph_aware(text, doc_id, analysis)
        
        # 第三層：精細分割
        return self._process_fine_split(text, doc_id, analysis)
    
    def _split_by_chapters(self, text: str) -> List[str]:
        """按章節分割文本"""
        chapter_patterns = [
            r'\n(第[一二三四五六七八九十\d]+章[^\n]*)',
            r'\n(Chapter\s+\d+[^\n]*)',
            r'\n(\d+\.\s*[^\n]{5,50})\n',
            r'\n([A-Z][^\n]{10,50})\n(?=[A-Z]|$)'
        ]
        
        best_split = None
        best_count = 0
        
        for pattern in chapter_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            
            if len(matches) >= 2:
                splits = []
                start = 0
                
                for match in matches:
                    if start < match.start():
                        splits.append(text[start:match.start()].strip())
                    start = match.start()
                
                splits.append(text[start:].strip())
                splits = [s for s in splits if s and len(s) > 50]
                
                if len(splits) > best_count:
                    best_split = splits
                    best_count = len(splits)
        
        return best_split if best_split else [text]
    
    def _split_by_sections(self, text: str) -> List[str]:
        """按小節分割文本"""
        section_patterns = [
            r'\n(第[一二三四五六七八九十\d]+節[^\n]*)',
            r'\n(\d+\.\d+[^\n]*)',
            r'\n([一二三四五六七八九十]、[^\n]*)',
            r'\n([A-Z]\.[^\n]*)'
        ]
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            
            if len(matches) >= 2:
                splits = []
                start = 0
                
                for match in matches:
                    if start < match.start():
                        splits.append(text[start:match.start()].strip())
                    start = match.start()
                
                splits.append(text[start:].strip())
                return [s for s in splits if s and len(s) > 30]
        
        return [text]
    
    def _add_overlap_to_documents(self, documents: List[Document], overlap_size: int) -> List[Document]:
        """為文檔添加重疊部分"""
        if len(documents) <= 1:
            return documents
        
        for i in range(1, len(documents)):
            # 從前一個文檔末尾取重疊內容
            prev_content = documents[i-1].page_content
            current_content = documents[i].page_content
            
            if len(prev_content) > overlap_size:
                overlap_text = prev_content[-overlap_size:]
                # 尋找完整的句子邊界
                sentence_end = overlap_text.rfind('。')
                if sentence_end == -1:
                    sentence_end = overlap_text.rfind('.')
                
                if sentence_end > overlap_size // 2:
                    overlap_text = overlap_text[sentence_end+1:]
                
                documents[i].page_content = overlap_text + "\n" + current_content
                documents[i].metadata['has_overlap'] = True
                documents[i].metadata['overlap_length'] = len(overlap_text)
        
        return documents
    
    def _create_chunk_document(self, content: str, doc_id: str, chunk_index: int, 
                          analysis: TextAnalysis, split_method: str) -> Document:
        """創建分塊文檔 - 統一元數據格式"""
        token_count = self.token_estimator.estimate_tokens(content)
        
        # 搜尋URL
        url_regex = r'https?://[\w\-./?#&%=]+'
        found_urls = re.findall(url_regex, content)

        # 基本元數據（確保都是簡單類型）
        metadata = {
            'doc_id': str(doc_id),
            'chunk_id': f"{doc_id}_{chunk_index+1:03d}",
            'chunk_index': int(chunk_index),
            'text_type': str(analysis.text_type),
            'processing_strategy': str(analysis.processing_strategy),
            'split_method': str(split_method),
            'chunk_length': int(len(content)),
            'token_count': int(token_count),
            'quality_score': float(analysis.quality_score),
            'language': str(analysis.language),
            'has_overlap': False
        }

        # 🎯 正確方案：URL列表 → 分隔符字串 (這是唯一可行的方案)
        if found_urls:
            metadata['contained_urls'] = '|'.join(found_urls)  # ✅ 字符串類型
            metadata['url_count'] = len(found_urls)           # ✅ 整數類型
            metadata['has_urls'] = True                       # ✅ 布林類型
        else:
            metadata['contained_urls'] = ''                   # ✅ 空字符串
            metadata['url_count'] = 0                        # ✅ 整數 0
            metadata['has_urls'] = False                     # ✅ 布林 False

        # 🎯 正確方案：複雜結構 → JSON字串
        if analysis.structure_info:
            metadata['structure_info'] = json.dumps(analysis.structure_info, ensure_ascii=False)  # ✅ JSON字符串
            # 同時保留關鍵信息作為簡單字段（便於查詢）
            metadata['has_chapters'] = bool(analysis.structure_info.get('has_chapters', False))    # ✅ 布林
            metadata['has_sections'] = bool(analysis.structure_info.get('has_sections', False))    # ✅ 布林  
            metadata['paragraph_count'] = int(analysis.structure_info.get('paragraphs', 0))       # ✅ 整數
            metadata['sentence_count'] = int(analysis.structure_info.get('sentences', 0))         # ✅ 整數
        else:
            metadata['structure_info'] = '{}'        # ✅ 空JSON字符串
            metadata['has_chapters'] = False         # ✅ 布林
            metadata['has_sections'] = False         # ✅ 布林
            metadata['paragraph_count'] = 0          # ✅ 整數
            metadata['sentence_count'] = 0           # ✅ 整數
        
        return Document(page_content=content, metadata=metadata)