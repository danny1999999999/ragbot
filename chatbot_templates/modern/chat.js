document.addEventListener('DOMContentLoaded', () => {
    // Safari 移動端修復
    function setVH() {
        let vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${vh}px`);
    }

    function isMobileSafari() {
        return /iPad|iPhone|iPod/.test(navigator.userAgent);
    }

    if (window.innerWidth <= 768 || isMobileSafari()) {
        setVH();
        
        window.addEventListener('resize', setVH);
        window.addEventListener('orientationchange', () => {
            setTimeout(setVH, 500);
        });
        
        const chatInput = document.getElementById('chat-input');
        if (chatInput) {
            chatInput.addEventListener('focus', () => {
                setTimeout(setVH, 300);
            });
        }
    }
    
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
    let conversationHistory = [];
    let sessionId = localStorage.getItem('chat_session_id');
    if (!sessionId) {
        sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
        localStorage.setItem('chat_session_id', sessionId);
    }

    // 📝 新方案：簡化的文本渲染器 - 完全信任後端格式化
    class SimpleTextRenderer {
        constructor(element, htmlContent, onComplete) {
            this.element = element;
            this.htmlContent = htmlContent; // 直接接收HTML，不做任何修改
            this.onComplete = onComplete;
            this.tokens = [];
            this.currentIndex = 0;
            this.timeoutId = null;
            this.isRunning = false;
            this.speed = 30; // 打字速度
        }

        // 解析HTML為打字tokens
        parseHtml() {
            this.tokens = [];
            
            // 檢查是否包含參考鏈接區塊
            const referencePattern = /(\n\n💡 .*[\s\S]*)/;
            const hasReferences = referencePattern.test(this.htmlContent);
            
            if (hasReferences) {
                // 分離主要內容和參考區塊
                const parts = this.htmlContent.split(referencePattern);
                
                // 主要內容逐字打字
                if (parts[0]) {
                    this.parseContentForTyping(parts[0]);
                }
                
                // 參考區塊整體顯示
                if (parts[1]) {
                    this.tokens.push({
                        type: 'html_block',
                        content: parts[1],
                        instantShow: true
                    });
                }
            } else {
                // 沒有參考區塊，全部逐字打字
                this.parseContentForTyping(this.htmlContent);
            }
        }

        parseContentForTyping(content) {
            // 直接按段落分解，保持HTML結構
            const processedContent = this.preprocessForTyping(content);
            
            // 簡化邏輯：按HTML標籤和文本分離
            const htmlTagRegex = /<[^>]+>/g;
            let lastIndex = 0;
            let match;
            
            while ((match = htmlTagRegex.exec(processedContent)) !== null) {
                // 添加標籤前的文本
                const textBefore = processedContent.substring(lastIndex, match.index);
                if (textBefore) {
                    this.addTextTokens(textBefore);
                }
                
                // 添加HTML標籤（立即顯示）
                this.tokens.push({
                    type: 'html',
                    content: match[0],
                    instantShow: true
                });
                
                lastIndex = match.index + match[0].length;
            }
            
            // 添加最後剩餘的文本
            const remainingText = processedContent.substring(lastIndex);
            if (remainingText) {
                this.addTextTokens(remainingText);
            }
        }

        preprocessForTyping(content) {
            // 🔥 核心改變：只做最基本的HTML轉換，不修改任何換行
            let processed = content;
            
            // 處理 markdown 鏈接
            processed = processed.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, 
                '<a href="$2" target="_blank" rel="noopener noreferrer" class="source-link">$1</a>');
            
            // 處理粗體
            processed = processed.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            
            // 轉換換行為HTML - 保持原有格式
            processed = processed.replace(/\n/g, '<br>');
            
            return processed;
        }

        addTextTokens(text) {
            if (!text) return;
            
            // 逐字符添加
            for (let i = 0; i < text.length; i++) {
                const char = text[i];
                if (char.trim()) { // 跳過純空白字符的打字效果
                    const isPunctuation = /[。！？，、：；「」『』（）]/.test(char);
                    this.tokens.push({
                        type: 'text',
                        content: char,
                        instantShow: false,
                        pauseAfter: isPunctuation ? 100 : 0
                    });
                } else {
                    // 空白字符立即顯示
                    this.tokens.push({
                        type: 'text',
                        content: char,
                        instantShow: true
                    });
                }
            }
        }

        start() {
            if (this.isRunning) return;
            
            this.isRunning = true;
            isTyping = true;
            this.element.classList.remove('thinking');
            this.element.innerHTML = '';
            
            this.parseHtml();
            this.typeNextToken();
        }

        typeNextToken() {
            if (this.currentIndex >= this.tokens.length) {
                this.complete();
                return;
            }

            const token = this.tokens[this.currentIndex];
            this.currentIndex++;

            if (token.type === 'html' || token.type === 'html_block') {
                // HTML 內容直接插入
                this.element.insertAdjacentHTML('beforeend', token.content);
                
                // 綁定鏈接事件
                if (token.content.includes('class="source-link"')) {
                    this.bindLinkEvents();
                }
            } else {
                // 文字內容
                const textNode = document.createTextNode(token.content);
                this.element.appendChild(textNode);
            }

            scrollToBottom();

            const delay = token.instantShow ? 0 : this.speed;
            const pauseAfter = token.pauseAfter || 0;
            
            this.timeoutId = setTimeout(() => {
                this.typeNextToken();
            }, delay + pauseAfter);
        }

        bindLinkEvents() {
            const newLinks = this.element.querySelectorAll('.source-link:not([data-bound])');
            newLinks.forEach((link) => {
                link.setAttribute('data-bound', 'true');
                
                link.addEventListener('click', () => {
                    console.log('用戶點擊了參考鏈接:', link.href);
                    
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
        }

        complete() {
            this.isRunning = false;
            isTyping = false;
            
            this.bindLinkEvents();
            
            if (this.element.innerHTML.includes('💡 您可能想了解')) {
                this.element.classList.add('has-references');
            }
            
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
        }
    }

    // 主要函數
    const handleSendMessage = async () => {
        console.log("執行簡化版 chat.js - 信任後端格式化");
        const message = chatInput.value.trim();
        if (!message || isLoading || isTyping) return;

        // 添加到對話歷史
        conversationHistory.push({ role: 'user', content: message });

        addMessage(message, 'user');
        chatInput.value = '';
        updateUIState();

        const botMessageElement = addMessage('', 'bot', { isThinking: true });
        setLoading(true);

        try {
            const historyToSend = conversationHistory.slice(-10);

            const response = await fetch('api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    message, 
                    history: historyToSend,
                    session_id: sessionId,
                    format_for_frontend: true // 🔥 新增：告訴後端要預格式化
                })
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: '請求失敗' }));
                throw new Error(errorData.detail);
            }

            const data = await response.json();

            conversationHistory.push({ role: 'assistant', content: data.response });
            
            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }
            
            // 🔥 關鍵：直接使用後端返回的內容，不做任何修改
            renderBotMessageWithTyping(botMessageElement, data.response, () => {
                displayRecommendedQuestions(data.recommended_questions || []);
            });

        } catch (error) {
            renderBotMessage(botMessageElement, `🚫 錯誤: ${error.message}`);
        } finally {
            setLoading(false);
        }
    };

    // 🔥 簡化的渲染函數 - 完全信任後端
    const renderBotMessageWithTyping = (element, htmlContent, onComplete) => {
        console.log('🎭 接收後端格式化內容，直接渲染:', htmlContent.substring(0, 200));
        
        // 停止任何正在進行的打字效果
        if (currentTypingTimeout) {
            currentTypingTimeout.stop();
        }
        
        // 使用簡化渲染器
        currentTypingTimeout = new SimpleTextRenderer(element, htmlContent, onComplete);
        currentTypingTimeout.start();
    };

    // 即時渲染函數（用於錯誤訊息等）
    const renderBotMessage = (element, htmlContent) => {
        element.classList.remove('thinking');
        
        // 🔥 完全信任內容，不做任何修改
        element.innerHTML = htmlContent;
        
        // 綁定鏈接事件
        const links = element.querySelectorAll('.source-link');
        links.forEach((link, index) => {
            link.addEventListener('click', (e) => {
                console.log(`用戶點擊了參考鏈接 ${index + 1}:`, link.href);
                
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
        
        if (htmlContent.includes('💡 您可能想了解')) {
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

    const displayRecommendedQuestions = (questions) => {
        console.log('📄 處理推薦問題，數量:', questions ? questions.length : 0);
        
        const container = document.getElementById('recommended-questions-container');
        container.innerHTML = '';
        
        if (questions && questions.length > 0) {
            console.log('✅ 顯示推薦問題');
            
            let content = '<br>';
            content += '<div>';
            questions.forEach(q => {
                content += `<button onclick="document.getElementById('chat-input').value='${q.replace(/'/g, "\\'")}'; document.querySelector('#send-button').click();" style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 20px; padding: 8px 16px; margin: 4px; cursor: pointer; font-size: 14px;">${q}</button>`;
            });
            content += '</div>';
            
            container.innerHTML = content;
            console.log('✅ 推薦問題已顯示');
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
        const len = chatInput.value.length;
        charCounter.textContent = `${len}/2000`;
        charCounter.className = 'char-counter';
        if (len > 2000) charCounter.classList.add('danger');
        else if (len > 1800) charCounter.classList.add('warning');

        sendButton.disabled = len === 0 || len > 2000 || isLoading || isTyping;

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

    const errorClose = document.getElementById('error-close');
    if (errorClose) {
        errorClose.addEventListener('click', () => {
            document.getElementById('error-toast').style.display = 'none';
        });
    }

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

    console.log('✨ 聊天機器人界面初始化完成 - 簡化版');
    console.log('會話ID:', sessionId);
    console.log('🔥 完全信任後端格式化 - 不做任何文本修改');
});