document.addEventListener('DOMContentLoaded', () => {
    // Element references
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const charCounter = document.getElementById('char-counter');
    const themeSelectors = document.querySelectorAll('.theme-selector');
    const fontSizeSelector = document.getElementById('font-size-selector');

    // State
    let isLoading = false;
    let isTyping = false;
    let currentTypingTimeout = null;
    let conversationHistory = []; // ✨ 新增：用來儲存對話歷史
    let sessionId = localStorage.getItem('chat_session_id');
    if (!sessionId) {
        sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
        localStorage.setItem('chat_session_id', sessionId);
    }

    // 打字效果配置
    const TYPING_CONFIG = {
        speed: 30,
        enableCursor: true,
        pauseAtPunctuation: 100,
        pauseAtLineBreak: 200,
        instantTags: ['link', 'strong', 'bold']
    };

    // 智能文本解析器類
    class SmartTextParser {
        constructor(htmlText) {
            this.originalText = htmlText;
            this.tokens = [];
            this.parseText();
        }

        parseText() {
            let text = this.originalText;
            text = this.preprocessMarkdownLinks(text);
            text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            this.parseHtmlAndText(text);
        }

        preprocessMarkdownLinks(text) {
            const linkRegex = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g;
            return text.replace(linkRegex, (match, linkText, url) => {
                return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="source-link">${linkText}</a>`;
            });
        }

        parseHtmlAndText(htmlText) {
            const sections = this.splitIntoSections(htmlText);
            
            sections.forEach(section => {
                if (section.type === 'reference_block') {
                    // "您可能想知道"區塊作為整體處理
                    this.tokens.push({
                        type: 'html',
                        content: section.content,
                        instantShow: true
                    });
                } else if (section.type === 'list_item') {
                    // 數字列表項逐字打字
                    this.addTextTokens(section.content);
                } else if (section.type === 'paragraph') {
                    // 普通段落逐字打字
                    this.addTextTokens(section.content);
                } else if (section.type === 'html') {
                    // HTML 標籤整體處理
                    this.tokens.push({
                        type: 'html',
                        content: section.content,
                        instantShow: true
                    });
                }
            });
        }

        splitIntoSections(text) {
            const sections = [];
            
            // 🔧 關鍵修正：先用正則表達式分割出參考區塊
            const referencePattern = /(\n\n💡 你可能想了解[\s\S]*)/;
            const parts = text.split(referencePattern);
            
            for (let i = 0; i < parts.length; i++) {
                const part = parts[i]; // 關鍵修正：移除 .trim()
                if (!part) continue;
                
                if (part.includes('💡 你可能想了解')) {
                    // 參考區塊整體處理
                    sections.push({
                        type: 'reference_block',
                        content: part
                    });
                } else {
                    // 普通內容按原邏輯處理
                    const lines = part.split('\n');
                    let currentSection = '';
                    let currentType = 'paragraph';
                    
                    for (let j = 0; j < lines.length; j++) {
                        const line = lines[j].trim();
                        
                        // 檢測數字列表項
                        if (/^\d+\.\s*/.test(line)) {
                            if (currentSection.trim()) {
                                sections.push({ type: currentType, content: currentSection.trim() });
                            }
                            
                            let listContent = line;
                            if (j + 1 < lines.length) {
                                const nextLine = lines[j + 1].trim();
                                if (nextLine.startsWith('：') || nextLine.startsWith(':')) {
                                    listContent += ' ' + nextLine;
                                    j++;
                                }
                            }
                            
                            sections.push({ type: 'list_item', content: listContent });
                            currentSection = '';
                            currentType = 'paragraph';
                            continue;
                        }

                        // 檢測 HTML 標籤
                        if (line.includes('<') && line.includes('>')) {
                            if (currentSection.trim()) {
                                sections.push({ type: currentType, content: currentSection.trim() });
                                currentSection = '';
                            }
                            sections.push({ type: 'html', content: line });
                            continue;
                        }

                        // 累積段落內容
                        if (line) {
                            currentSection = currentSection ? currentSection + '\n' + line : line;
                        } else if (currentSection) {
                            sections.push({ type: currentType, content: currentSection.trim() });
                            currentSection = '';
                        }
                    }

                    if (currentSection.trim()) {
                        sections.push({ type: currentType, content: currentSection.trim() });
                    }
                }
            }

            return sections;
        }

        // 🔧 修復後的 addTextTokens 方法 - 支援行內 HTML 標籤
        addTextTokens(text) {
            if (!text) return;
            
            const lines = text.split('\n');
            
            lines.forEach((line, lineIndex) => {
                if (line) {
                    // 檢查是否是"你可能想了解"標題行
                    if (line.includes('💡 你可能想了解')) {
                        // 標題行直接添加為一個整體
                        this.tokens.push({
                            type: 'text',
                            content: line,
                            instantShow: false,
                            pauseAfter: 200
                        });
                    } else {
                        // 🔧 新增：解析行內 HTML 標籤
                        this.parseLineWithHtmlTags(line);
                    }
                }
                
                // 添加換行符（除了最後一行）
                if (lineIndex < lines.length - 1) {
                    this.tokens.push({
                        type: 'linebreak',
                        content: '<br>',
                        instantShow: true,
                        pauseAfter: TYPING_CONFIG.pauseAtLineBreak
                    });
                }
            });
        }
        
        // 🔧 新增：解析含有 HTML 標籤的行
        parseLineWithHtmlTags(line) {
            // 定義要處理的 HTML 標籤
            const htmlTagPattern = /(<\/?(strong|b|em|i|u)>)/gi;
            
            let lastIndex = 0;
            let match;
            
            // 重設正則表達式的 lastIndex
            htmlTagPattern.lastIndex = 0;
            
            // 使用正則表達式找到所有 HTML 標籤
            while ((match = htmlTagPattern.exec(line)) !== null) {
                // 添加標籤前的文字（逐字打字）
                const textBefore = line.substring(lastIndex, match.index);
                if (textBefore) {
                    this.addTextCharByChar(textBefore);
                }
                
                // 添加 HTML 標籤（瞬間顯示）
                this.tokens.push({
                    type: 'html',
                    content: match[1],
                    instantShow: true,
                    pauseAfter: 0
                });
                
                lastIndex = match.index + match[1].length;
            }
            
            // 添加最後剩餘的文字（逐字打字）
            const remainingText = line.substring(lastIndex);
            if (remainingText) {
                this.addTextCharByChar(remainingText);
            }
        }
        
        // 🔧 新增：逐字添加文字的輔助方法
        addTextCharByChar(text) {
            for (let i = 0; i < text.length; i++) {
                const char = text[i];
                const isPunctuation = /[。！？，、：；「」『』（）]/.test(char);
                
                this.tokens.push({
                    type: 'text',
                    content: char,
                    instantShow: false,
                    pauseAfter: isPunctuation ? TYPING_CONFIG.pauseAtPunctuation : 0
                });
            }
        }

        getTokens() {
            return this.tokens;
        }
    }

    // 智能打字效果渲染器
    class TypingRenderer {
        constructor(element, tokens, onComplete) {
            this.element = element;
            this.tokens = tokens;
            this.onComplete = onComplete;
            this.currentIndex = 0;
            this.timeoutId = null;
            this.isRunning = false;
        }

        start() {
            if (this.isRunning) return;
            
            this.isRunning = true;
            isTyping = true;
            this.element.classList.remove('thinking');
            this.element.innerHTML = '';
            
            // 添加打字游標
            if (TYPING_CONFIG.enableCursor) {
                this.cursor = document.createElement('span');
                this.cursor.className = 'typing-cursor';
                this.cursor.textContent = '|';
                this.cursor.style.cssText = `
                    animation: blink 1s infinite;
                    color: #007bff;
                    font-weight: bold;
                `;
                this.element.appendChild(this.cursor);
                this.addCursorAnimation();
            }
            
            this.typeNextToken();
        }

        typeNextToken() {
            if (this.currentIndex >= this.tokens.length) {
                this.complete();
                return;
            }

            const token = this.tokens[this.currentIndex];
            this.currentIndex++;

            // 移除游標
            if (this.cursor && this.cursor.parentNode) {
                this.cursor.remove();
            }

            // 添加內容
            if (token.type === 'html' || token.type === 'linebreak') {
                // HTML 標籤和換行符一次性添加 
                this.element.insertAdjacentHTML('beforeend', token.content);
                
                // 如果是連結，立即綁定事件
                if (token.content.includes('class="source-link"')) {
                    this.bindLinkEvents();
                }
            } else {
                // 普通文字逐字添加 
                const textNode = document.createTextNode(token.content);
                this.element.appendChild(textNode);
            }

            // 重新添加游標
            if (TYPING_CONFIG.enableCursor && this.currentIndex < this.tokens.length) {
                this.element.appendChild(this.cursor);
            }

            // 滾動到底部
            scrollToBottom();

            // 設定下一個字符的延遲
            const delay = token.instantShow ? 0 : TYPING_CONFIG.speed;
            const pauseAfter = token.pauseAfter || 0;
            
            this.timeoutId = setTimeout(() => {
                this.typeNextToken();
            }, delay + pauseAfter);
        }

        bindLinkEvents() {
            // 為新添加的連結綁定事件
            const newLinks = this.element.querySelectorAll('.source-link:not([data-bound])');
            newLinks.forEach((link) => {
                link.setAttribute('data-bound', 'true');
                
                link.addEventListener('click', () => {
                    console.log('使用者點擊了參考連結:', link.href);
                    console.log('連結標題:', link.textContent);
                    
                    try {
                        fetch('/api/link_click', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                url: link.href,
                                title: link.textContent,
                                timestamp: new Date().toISOString(),
                                session_id: sessionId
                            })
                        }).catch(err => console.log('統計記錄失敗:', err));
                    } catch (err) {
                        console.log('統計記錄異常:', err);
                    }
                });
                
                // 添加懸停效果
                link.addEventListener('mouseenter', () => {
                    link.style.transform = 'translateY(-1px)';
                });
                
                link.addEventListener('mouseleave', () => {
                    link.style.transform = 'translateY(0)';
                });
            });
        }

        complete() {
            this.isRunning = false;
            isTyping = false;
            
            // 移除游標
            if (this.cursor && this.cursor.parentNode) {
                this.cursor.remove();
            }
            
            // 確保所有連結都有事件綁定
            this.bindLinkEvents();
            
            // 確保"你可能想了解"區塊有正確的樣式
            if (this.element.innerHTML.includes('💡 你可能想了解')) {
                this.element.classList.add('has-references');
            }
            
            // 執行完成回調
            if (this.onComplete) {
                this.onComplete();
            }
        }

        stop() {
            if (this.timeoutId) {
                clearTimeout(this.timeoutId);
                this.timeoutId = null;
            }
            this.isRunning = false;
            isTyping = false;
            
            if (this.cursor && this.cursor.parentNode) {
                this.cursor.remove();
            }
        }

        addCursorAnimation() {
            // 動態添加游標動畫 CSS
            if (!document.getElementById('typing-cursor-style')) {
                const style = document.createElement('style');
                style.id = 'typing-cursor-style';
                style.textContent = `
                    @keyframes blink {
                        0%, 50% { opacity: 1; }
                        51%, 100% { opacity: 0; }
                    }
                `;
                document.head.appendChild(style);
            }
        }
    }

    // 主要函數
    const handleSendMessage = async () => {
        const message = chatInput.value.trim();
        if (!message || isLoading || isTyping) return;

        // ✨ 步驟 1: 將用戶訊息添加到歷史紀錄
        conversationHistory.push({ role: 'user', content: message });

        // Add user message to UI
        addMessage(message, 'user');
        chatInput.value = '';
        updateUIState();

        // Show thinking indicator
        const botMessageElement = addMessage('', 'bot', { isThinking: true });
        setLoading(true);

        try {
            // ✨ 步驟 2: 發送包含歷史紀錄的請求
            // 我們只發送最近10則訊息以避免請求過大
            const historyToSend = conversationHistory.slice(-10);

            const response = await fetch('api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message, 
                    history: historyToSend, // ✨ 發送歷史
                    session_id: sessionId 
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: '請求失敗' }));
                throw new Error(errorData.detail);
            }

            const data = await response.json();

            // ✨ 步驟 3: 將機器人回應添加到歷史紀錄
            conversationHistory.push({ role: 'assistant', content: data.response });

            // ✨ 步驟 4: 管理歷史紀錄長度，避免無限增長
            if (conversationHistory.length > 20) { // 保留最近的20則訊息
                conversationHistory = conversationHistory.slice(-20);
            }
            
            // 使用智能打字效果渲染回應
            renderBotMessageWithTyping(botMessageElement, data.response, () => {
                displayRecommendedQuestions(data.recommended_questions || []);
            });

        } catch (error) {
            // 錯誤消息不使用打字效果
            renderBotMessage(botMessageElement, `🚫 錯誤: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };

    // 帶打字效果的機器人回應渲染
    const renderBotMessageWithTyping = (element, text, onComplete) => {
        // 解析文本
        const parser = new SmartTextParser(text);
        const tokens = parser.getTokens();
        
        // 停止任何正在進行的打字效果
        if (currentTypingTimeout) {
            currentTypingTimeout.stop();
        }
        
        // 開始新的打字效果
        currentTypingTimeout = new TypingRenderer(element, tokens, onComplete);
        currentTypingTimeout.start();
    };

    // 即時渲染函數（用於錯誤消息等）
    const renderBotMessage = (element, text) => {
        element.classList.remove('thinking');
        
        // 改進的連結渲染邏輯
        let processedText = text;
        
        // 處理markdown格式的連結: [文本](URL)
        const linkRegex = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g;
        processedText = processedText.replace(linkRegex, (match, linkText, url) => {
            return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="source-link">${linkText}</a>`;
        });
        
        // 處理**加粗**文本
        processedText = processedText.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        // 處理換行
        processedText = processedText.replace(/\n/g, '<br>');
        
        // 安全地設置HTML內容
        element.innerHTML = processedText;
        
        // 為連結添加點擊事件和統計
        const links = element.querySelectorAll('.source-link');
        links.forEach((link, index) => {
            link.addEventListener('click', (e) => {
                console.log(`使用者點擊了參考連結 ${index + 1}:`, link.href);
                console.log('連結標題:', link.textContent);
                
                try {
                    fetch('/api/link_click', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            url: link.href,
                            title: link.textContent,
                            timestamp: new Date().toISOString(),
                            session_id: sessionId
                        })
                    }).catch(err => console.log('統計記錄失敗:', err));
                } catch (err) {
                    console.log('統計記錄異常:', err);
                }
            });
            
            link.addEventListener('mouseenter', () => {
                link.style.transform = 'translateY(-1px)';
            });
            
            link.addEventListener('mouseleave', () => {
                link.style.transform = 'translateY(0)';
            });
        });
        
        if (processedText.includes('💡 你可能想了解')) {
            element.classList.add('has-references');
        }
    };

    const addMessage = (text, type, options = {}) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${type}-message`);

        if (options.isThinking) {
            messageElement.classList.add('thinking');
        } else {
            messageElement.textContent = text;
        }

        chatMessages.appendChild(messageElement);
        scrollToBottom();
        return messageElement;
    };

    // ✅ 完全替換為這個修正版本：
    const displayRecommendedQuestions = (questions) => {
        console.log('🔄 處理推薦問題，數量:', questions ? questions.length : 0);
        
        const container = document.getElementById('recommended-questions-container');
        container.innerHTML = '';
        
        // 🔧 修正：只處理推薦問題，完全不檢測參考連結
        if (questions && questions.length > 0) {
            console.log('✅ 顯示推薦問題');
            
            let content = '<br>';
            
            // 只顯示推薦問題按鈕，不添加任何標題
            content += '<div>';
            questions.forEach(q => {
                content += `<button onclick="document.getElementById('chat-input').value='${q.replace(/'/g, "\\'")}'; document.querySelector('#send-button').click();" style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 20px; padding: 8px 16px; margin: 4px; cursor: pointer; font-size: 14px;">${q}</button>`;
            });
            content += '</div>';
            
            container.innerHTML = content;
            console.log('✅ 推薦問題已顯示');
        } else {
            console.log('❌ 沒有推薦問題，不顯示任何內容');
            // 🔧 重要：什麼都不顯示
        }
    };


    // UI 狀態管理
    const setLoading = (loadingState) => {
        isLoading = loadingState;
        sendButton.disabled = isLoading || isTyping;
        const sendIcon = sendButton.querySelector('.send-icon');
        const loadingIcon = sendButton.querySelector('.loading-icon');
        if (sendIcon) sendIcon.style.display = isLoading ? 'none' : 'inline';
        if (loadingIcon) loadingIcon.style.display = isLoading ? 'inline' : 'none';
    };

    const updateUIState = () => {
        // Char counter
        const len = chatInput.value.length;
        charCounter.textContent = `${len}/2000`;
        charCounter.className = 'char-counter';
        if (len > 2000) charCounter.classList.add('danger');
        else if (len > 1800) charCounter.classList.add('warning');

        // Send button
        sendButton.disabled = len === 0 || len > 2000 || isLoading || isTyping;

        // Textarea height
        chatInput.style.height = 'auto';
        chatInput.style.height = `${Math.min(chatInput.scrollHeight, 100)}px`;
    };

    const applyTheme = (theme) => {
        document.body.className = `theme-${theme}`;
        localStorage.setItem('chat_theme', theme);
        
        themeSelectors.forEach(selector => {
            selector.classList.toggle('active', selector.dataset.theme === theme);
        });
    };

    const applyFontSize = (size) => {
        document.documentElement.style.setProperty('--chat-font-size', `${size}px`);
        localStorage.setItem('chat_font_size', size);
    };

    const scrollToBottom = () => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    };
    
    const escapeHtml = (text) => {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    };

    // 事件監聽器
    sendButton.addEventListener('click', handleSendMessage);
    
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isTyping) {
                handleSendMessage();
            }
        }
    });
    
    chatInput.addEventListener('input', updateUIState);

    themeSelectors.forEach(selector => {
        selector.addEventListener('click', () => {
            applyTheme(selector.dataset.theme);
        });
    });

    fontSizeSelector.addEventListener('change', (e) => {
        const sizeMap = { small: 14, medium: 16, large: 18 };
        applyFontSize(sizeMap[e.target.value]);
    });

    // 錯誤提示關閉按鈕
    const errorClose = document.getElementById('error-close');
    if (errorClose) {
        errorClose.addEventListener('click', () => {
            document.getElementById('error-toast').style.display = 'none';
        });
    }

    // 阻止在打字期間的意外操作
    document.addEventListener('keydown', (e) => {
        if (isTyping && (e.key === 'Enter' || e.key === 'Escape')) {
            e.preventDefault();
        }
    });

    // 初始化
    const savedTheme = localStorage.getItem('chat_theme') || 'blue';
    const savedFontSize = localStorage.getItem('chat_font_size') || '16';
    const savedFontSizeKey = Object.keys({ small: 14, medium: 16, large: 18 }).find(
        key => ({ small: 14, medium: 16, large: 18 }[key] == savedFontSize)
    ) || 'medium';

    applyTheme(savedTheme);
    applyFontSize(savedFontSize);
    fontSizeSelector.value = savedFontSizeKey;

    updateUIState();
    chatInput.focus();

    // 初始化完成日誌
    console.log('聊天機器人界面初始化完成');
    console.log('會話ID:', sessionId);
    console.log('智能打字效果已啟用');
});