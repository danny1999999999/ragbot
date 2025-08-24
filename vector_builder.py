#!/usr/bin/env python3
"""
向量建立層 - 自適應批次處理
- 🧠 智能批次大小調整
- 📊 性能統計與監控
- 🔧 大文檔自動分割
- ⚡ 自適應處理策略
"""

from core_config import *
from text_processing import AdvancedTokenEstimator
import time
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from langchain_core.documents import Document

class AdaptiveBatchProcessor:
    """🔧 自適應批次處理器"""
    
    def __init__(self):
        self.token_estimator = AdvancedTokenEstimator()
        self.max_tokens_per_batch = TOKEN_LIMITS["max_tokens_per_request"]
        self.max_batch_size = TOKEN_LIMITS["max_batch_size"]
        self.adaptive_batching = TOKEN_LIMITS.get("adaptive_batching", True)
        
        # 性能統計
        self.batch_stats = {
            'total_batches': 0,
            'total_documents': 0,
            'total_tokens': 0,
            'avg_batch_time': 0,
            'success_rate': 0,
            'adaptive_adjustments': 0
        }
        
        # 自適應參數
        self.current_batch_size = self.max_batch_size
        self.success_streak = 0
        self.failure_streak = 0
    
    def create_smart_batches(self, documents: List[Document]) -> List[Tuple[List[Document], Dict]]:
        """創建智能批次"""
        if not documents:
            return []
        
        print(f"🔧 創建智能批次...")
        print(f"   📄 總文檔數: {len(documents)}")
        print(f"   🔢 Token 限制: {self.max_tokens_per_batch:,}")
        print(f"   📦 最大批次大小: {self.max_batch_size}")
        print(f"   🧠 自適應批次: {'✅' if self.adaptive_batching else '❌'}")
        
        batches = []
        current_batch = []
        current_tokens = 0
        batch_info = {'documents': 0, 'tokens': 0, 'types': defaultdict(int)}
        
        # 按 token 數排序文檔（大的在前面，更容易處理）
        sorted_docs = sorted(
            documents, 
            key=lambda doc: doc.metadata.get('token_count', 
                self.token_estimator.estimate_tokens(doc.page_content)),
            reverse=True
        )
        
        for doc_idx, doc in enumerate(sorted_docs):
            doc_tokens = doc.metadata.get('token_count') or \
                        self.token_estimator.estimate_tokens(doc.page_content)
            
            # 檢查單個文檔是否過大
            if doc_tokens > self.max_tokens_per_batch:
                print(f"   ⚠️ 文檔 {doc_idx+1} 過大 ({doc_tokens:,} tokens)，需要分割")
                
                # 完成當前批次
                if current_batch:
                    batches.append((current_batch, dict(batch_info)))
                    current_batch = []
                    current_tokens = 0
                    batch_info = {'documents': 0, 'tokens': 0, 'types': defaultdict(int)}
                
                # 分割大文檔
                split_docs = self._split_large_document(doc)
                for split_doc in split_docs:
                    split_tokens = self.token_estimator.estimate_tokens(split_doc.page_content)
                    split_doc.metadata['token_count'] = split_tokens
                    batches.append(([split_doc], {
                        'documents': 1, 
                        'tokens': split_tokens, 
                        'types': {split_doc.metadata.get('text_type', 'unknown'): 1},
                        'is_split': True
                    }))
                continue
            
            # 檢查是否可以加入當前批次
            would_exceed_tokens = current_tokens + doc_tokens > self.max_tokens_per_batch
            would_exceed_size = len(current_batch) >= self._get_adaptive_batch_size()
            
            if would_exceed_tokens or would_exceed_size:
                # 完成當前批次
                if current_batch:
                    batches.append((current_batch, dict(batch_info)))
                
                # 開始新批次
                current_batch = [doc]
                current_tokens = doc_tokens
                batch_info = {
                    'documents': 1, 
                    'tokens': doc_tokens, 
                    'types': defaultdict(int)
                }
                batch_info['types'][doc.metadata.get('text_type', 'unknown')] += 1
            else:
                # 加入當前批次
                current_batch.append(doc)
                current_tokens += doc_tokens
                batch_info['documents'] += 1
                batch_info['tokens'] += doc_tokens
                batch_info['types'][doc.metadata.get('text_type', 'unknown')] += 1
        
        # 處理最後一個批次
        if current_batch:
            batches.append((current_batch, dict(batch_info)))
        
        # 統計信息
        total_docs = sum(info['documents'] for _, info in batches)
        total_tokens = sum(info['tokens'] for _, info in batches)
        avg_tokens_per_batch = total_tokens / len(batches) if batches else 0
        
        print(f"✅ 智能批次創建完成:")
        print(f"   📦 總批次數: {len(batches)}")
        print(f"   📄 總文檔數: {total_docs}")
        print(f"   🔢 總 tokens: {total_tokens:,}")
        print(f"   📊 平均 tokens/批次: {avg_tokens_per_batch:.0f}")
        print(f"   💰 估算成本: ${self.token_estimator.estimate_embedding_cost(total_tokens):.4f}")
        
        # 更新統計
        self.batch_stats['total_batches'] += len(batches)
        self.batch_stats['total_documents'] += total_docs
        self.batch_stats['total_tokens'] += total_tokens
        
        return batches
    
    def _get_adaptive_batch_size(self) -> int:
        """獲取自適應批次大小"""
        if not self.adaptive_batching:
            return self.max_batch_size
        
        # 根據成功率調整批次大小
        if self.success_streak >= 3:
            # 連續成功，可以增加批次大小
            self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
            self.batch_stats['adaptive_adjustments'] += 1
        elif self.failure_streak >= 2:
            # 連續失敗，減少批次大小
            self.current_batch_size = max(1, self.current_batch_size - 2)
            self.batch_stats['adaptive_adjustments'] += 1
        
        return self.current_batch_size
    
    def _split_large_document(self, doc: Document) -> List[Document]:
        """分割過大的文檔"""
        content = doc.page_content
        max_chars = int(self.max_tokens_per_batch * 2.5)  # 估算字符數
        
        # 嘗試按段落分割
        paragraphs = content.split('\n\n')
        split_docs = []
        current_content = ""
        part_index = 0
        
        for para in paragraphs:
            if len(current_content + para) <= max_chars or not current_content:
                current_content += ("\n\n" if current_content else "") + para
            else:
                if current_content.strip():
                    split_doc = Document(
                        page_content=current_content,
                        metadata={
                            **doc.metadata,
                            'chunk_id': f"{doc.metadata.get('chunk_id', 'unknown')}_part{part_index+1}",
                            'is_split_part': True,
                            'part_index': part_index,
                            'split_reason': 'token_limit'
                        }
                    )
                    split_docs.append(split_doc)
                    part_index += 1
                current_content = para
        
        # 處理最後一部分
        if current_content.strip():
            split_doc = Document(
                page_content=current_content,
                metadata={
                    **doc.metadata,
                    'chunk_id': f"{doc.metadata.get('chunk_id', 'unknown')}_part{part_index+1}",
                    'is_split_part': True,
                    'part_index': part_index,
                    'split_reason': 'token_limit'
                }
            )
            split_docs.append(split_doc)
        
        return split_docs or [doc]  # 如果分割失敗，返回原文檔
    
    def record_batch_result(self, success: bool, processing_time: float = 0):
        """記錄批次處理結果"""
        if success:
            self.success_streak += 1
            self.failure_streak = 0
        else:
            self.failure_streak += 1
            self.success_streak = 0
        
        # 更新成功率
        total_attempts = self.batch_stats['total_batches']
        if total_attempts > 0:
            self.batch_stats['success_rate'] = (total_attempts - self.failure_streak) / total_attempts
        
        # 更新平均處理時間
        if processing_time > 0:
            current_avg = self.batch_stats['avg_batch_time']
            self.batch_stats['avg_batch_time'] = (current_avg + processing_time) / 2
    
    def get_performance_stats(self) -> Dict:
        """獲取性能統計"""
        return dict(self.batch_stats)