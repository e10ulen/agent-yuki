package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/bwmarrin/discordgo"
)

const (
	ollamaBaseURL = "http://localhost:11434"
	modelName     = "qwen3:0.6b" // 14bに変えるとさらに安定
)

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type OllamaChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
	Stream   bool      `json:"stream"`
}

type OllamaChatResponse struct {
	Model     string  `json:"model"`
	CreatedAt string  `json:"created_at"`
	Message   Message `json:"message"`
	Done      bool    `json:"done"`
}

type Tool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

type AllowedPaths struct {
	Dirs  []string `json:"dirs"`
	Files []string `json:"files"`
}

var searchWebTool = Tool{
	Type: "function",
	Function: struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	}{
		Name: "search_web",
		Description: `インターネット上の最新情報を SearXNG 経由で検索します。
日本語の質問にも対応しています。ニュース・技術情報・商品比較などに便利です。

主な引数:
- query     : 検索したいキーワードや質問文（必須）
- engines   : 使用したい検索エンジン（カンマ区切り。例: "google,duckduckgo,yandex,bing"）
- categories: 検索カテゴリ（例: "general", "news", "it", "science", "social media"）
- lang      : 言語（例: "ja", "en", "all"）
- safesearch: 0=無効, 1=中程度, 2=厳格（通常は1でOK）

返されるのはタイトル・URL・スニペットのリストです。`,
		Parameters: json.RawMessage(`{
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "検索クエリ（質問文でもOK）"
                },
                "engines": {
                    "type": "string",
                    "description": "使用エンジン（カンマ区切り、省略可）"
                },
                "categories": {
                    "type": "string",
                    "description": "カテゴリ（カンマ区切り、省略可）"
                },
                "lang": {
                    "type": "string",
                    "description": "言語コード（例: ja, en, all）",
                    "default": "ja"
                },
                "safesearch": {
                    "type": "integer",
                    "description": "セーフサーチレベル（0〜2）",
                    "default": 1
                }
            },
            "required": ["query"]
        }`),
	},
}

var (
	historyMu   sync.Mutex
	historyDir  = "chat_history"
	logDir      = "log"
	allowedMu   sync.Mutex
	allowedFile = "allowed_paths.json"
	allowedData = make(map[string]AllowedPaths)

	pythonTool    = Tool{ /* 既存の定義 */ }
	readFileTool  = Tool{ /* 既存 */ }
	listDirTool   = Tool{ /* 既存 */ }
	writeFileTool = Tool{ /* 既存 */ }
)

// loadAllowedPaths, saveAllowedPaths, loadHistory, saveHistory, callOllamaChat
// searchViaSearxNG, executePythonCode, isPathAllowedForWrite, readFileToolFunc, listDirToolFunc, writeFileToolFunc
// saveLog は変更なし（省略せず全部入れる）

// loadAllowedPaths
func loadAllowedPaths() {
	data, err := os.ReadFile(allowedFile)
	if err == nil {
		json.Unmarshal(data, &allowedData)
	}
}

// saveAllowedPaths
func saveAllowedPaths() {
	data, _ := json.MarshalIndent(allowedData, "", "  ")
	os.WriteFile(allowedFile, data, 0644)
}

// loadHistory
func loadHistory(channelID string) ([]Message, error) {
	historyMu.Lock()
	defer historyMu.Unlock()

	if err := os.MkdirAll(historyDir, 0755); err != nil {
		return nil, err
	}

	path := filepath.Join(historyDir, channelID+".json")
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return []Message{}, nil
	}
	if err != nil {
		return nil, err
	}

	var msgs []Message
	json.Unmarshal(data, &msgs)
	return msgs, nil
}

// saveHistory
func saveHistory(channelID string, msgs []Message) error {
	historyMu.Lock()
	defer historyMu.Unlock()

	path := filepath.Join(historyDir, channelID+".json")
	data, _ := json.MarshalIndent(msgs, "", "  ")
	return os.WriteFile(path, data, 0644)
}

// callOllamaChat
func callOllamaChat(messages []Message, tools []Tool) (*OllamaChatResponse, error) {
	req := OllamaChatRequest{
		Model:    modelName,
		Messages: messages,
		Tools:    tools,
		Stream:   false,
	}

	body, _ := json.Marshal(req)
	resp, err := http.Post(ollamaBaseURL+"/api/chat", "application/json", bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var chatResp OllamaChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, err
	}
	return &chatResp, nil
}

// searchViaSearxNG (省略せず全部)
type SearxNGResult struct {
	Title     string `json:"title"`
	URL       string `json:"url"`
	Content   string `json:"content"`
	Published string `json:"published,omitempty"`
	Age       string `json:"age,omitempty"`
}

type SearxNGResponse struct {
	Results         []SearxNGResult `json:"results"`
	NumberOfResults int             `json:"number_of_results"`
}

func searchViaSearxNG(query string, engines, categories, lang string, safesearch int) (string, error) {
	baseURL := os.Getenv("SEARXNG_URL")
	if baseURL == "" {
		return "", fmt.Errorf("環境変数 SEARXNG_URL が設定されていません")
	}

	reqBody := map[string]interface{}{
		"q":          query,
		"format":     "json",
		"lang":       lang,
		"safesearch": safesearch,
	}
	if engines != "" {
		reqBody["engines"] = engines
	}
	if categories != "" {
		reqBody["categories"] = categories
	}

	bodyBytes, _ := json.Marshal(reqBody)

	req, err := http.NewRequest("POST", baseURL+"/search", bytes.NewBuffer(bodyBytes))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 20 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("SearXNG がエラーを返しました: %d", resp.StatusCode)
	}

	var sr SearxNGResponse
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return "", err
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("【検索クエリ】 %s\n", query))
	sb.WriteString(fmt.Sprintf("結果件数: %d\n\n", sr.NumberOfResults))

	if len(sr.Results) == 0 {
		sb.WriteString("該当する結果が見つかりませんでした。\n")
		return sb.String(), nil
	}

	for i, r := range sr.Results {
		if i >= 10 {
			sb.WriteString("...（続きは省略）\n")
			break
		}
		sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, r.Title))
		sb.WriteString(fmt.Sprintf("   URL: %s\n", r.URL))
		if r.Published != "" || r.Age != "" {
			sb.WriteString(fmt.Sprintf("   公開/更新: %s %s\n", r.Published, r.Age))
		}
		sb.WriteString(fmt.Sprintf("   %s\n\n", strings.TrimSpace(r.Content)))
	}

	return sb.String(), nil
}

// executePythonCode (省略せず)
func executePythonCode(code string) string {
	cmd := exec.Command("python3", "-c", code)
	cmd.Env = []string{"PATH=/usr/bin:/bin:/usr/local/bin", "PYTHONPATH="}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return fmt.Sprintf("実行エラー: %v", err)
	}

	done := make(chan error)
	go func() { done <- cmd.Wait() }()

	select {
	case <-time.After(15 * time.Second):
		cmd.Process.Kill()
		return "タイムアウト（15秒超過）"
	case err := <-done:
		output := stdout.String()
		if err != nil {
			output += "\n--- エラー ---\n" + stderr.String()
		}
		if output == "" {
			return "(出力なし)"
		}
		return output
	}
}

// isPathAllowedForWrite
func isPathAllowedForWrite(targetPath, channelID string) bool {
	allowedMu.Lock()
	ap := allowedData[channelID]
	allowedMu.Unlock()

	targetAbs, err := filepath.Abs(targetPath)
	if err != nil {
		return false
	}

	for _, allowedDir := range ap.Dirs {
		allowedAbs, _ := filepath.Abs(allowedDir)
		if strings.HasPrefix(targetAbs, allowedAbs+string(filepath.Separator)) ||
			targetAbs == allowedAbs {
			return true
		}
	}

	return false
}

// readFileToolFunc
func readFileToolFunc(path, channelID string) string {
	allowedMu.Lock()
	ap := allowedData[channelID]
	allowedMu.Unlock()

	allowed := false
	for _, dir := range ap.Dirs {
		if strings.HasPrefix(path, dir) || path == dir {
			allowed = true
			break
		}
	}
	for _, f := range ap.Files {
		if path == f {
			allowed = true
			break
		}
	}

	if !allowed {
		return "エラー: このファイルは許可されていません。"
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "読み込みエラー: " + err.Error()
	}
	return string(data)
}

// listDirToolFunc
func listDirToolFunc(path, channelID string) string {
	allowedMu.Lock()
	ap := allowedData[channelID]
	allowedMu.Unlock()

	allowed := false
	for _, dir := range ap.Dirs {
		if path == dir || strings.HasPrefix(path, dir+string(filepath.Separator)) {
			allowed = true
			break
		}
	}
	if !allowed {
		return "エラー: このディレクトリは許可されていません。"
	}

	entries, err := os.ReadDir(path)
	if err != nil {
		return "ディレクトリ読み込みエラー: " + err.Error()
	}

	var sb strings.Builder
	sb.WriteString("ディレクトリ内容:\n")
	for _, entry := range entries {
		sb.WriteString("- " + entry.Name())
		if entry.IsDir() {
			sb.WriteString(" (dir)")
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// writeFileToolFunc
func writeFileToolFunc(path, content, channelID string) string {
	if !isPathAllowedForWrite(path, channelID) {
		return "エラー: このパスへの書き込みは許可されていません。!setdir で親ディレクトリを許可してください。"
	}

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "ディレクトリ作成エラー: " + err.Error()
	}

	err := os.WriteFile(path, []byte(content), 0644)
	if err != nil {
		return "書き込みエラー: " + err.Error()
	}

	return fmt.Sprintf("ファイル書き込み成功: %s\nサイズ: %d バイト", path, len(content))
}

// saveLog (2引数版)
func saveLog(query, result string) string {
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return ""
	}

	filename := fmt.Sprintf("agent_log_%s.md", time.Now().Format("20060102_150405"))
	path := filepath.Join(logDir, filename)

	summary := fmt.Sprintf(`# エージェント実行ログ

**指令**: %s

**実行日時**: %s

**結果**:
%s
`, query, time.Now().Format("2006-01-02 15:04:05 JST"), result)

	os.WriteFile(path, []byte(summary), 0644)
	return path
}

// messageHandler - ここが核心
func messageHandler(s *discordgo.Session, m *discordgo.MessageCreate) {
	if m.Author.ID == s.State.User.ID {
		return
	}

	content := strings.TrimSpace(m.Content)

	// 設定コマンド（省略せず全部）
	switch {
	case strings.HasPrefix(content, "!setdir "):
		dirPath := strings.TrimSpace(strings.TrimPrefix(content, "!setdir "))
		absPath, _ := filepath.Abs(dirPath)
		if _, err := os.Stat(absPath); os.IsNotExist(err) {
			s.ChannelMessageSend(m.ChannelID, "指定されたディレクトリが存在しません")
			return
		}

		allowedMu.Lock()
		ap := allowedData[m.ChannelID]
		ap.Dirs = append(ap.Dirs, absPath)
		allowedData[m.ChannelID] = ap
		allowedMu.Unlock()
		saveAllowedPaths()

		s.ChannelMessageSend(m.ChannelID, fmt.Sprintf("ディレクトリを許可しました: %s", absPath))
		return

	case strings.HasPrefix(content, "!addfile "):
		filePath := strings.TrimSpace(strings.TrimPrefix(content, "!addfile "))
		absPath, _ := filepath.Abs(filePath)
		if _, err := os.Stat(absPath); os.IsNotExist(err) {
			s.ChannelMessageSend(m.ChannelID, "指定されたファイルが存在しません")
			return
		}

		allowedMu.Lock()
		ap := allowedData[m.ChannelID]
		ap.Files = append(ap.Files, absPath)
		allowedData[m.ChannelID] = ap
		allowedMu.Unlock()
		saveAllowedPaths()

		s.ChannelMessageSend(m.ChannelID, fmt.Sprintf("ファイルを許可しました: %s", absPath))
		return

	case content == "!listdir":
		allowedMu.Lock()
		ap := allowedData[m.ChannelID]
		allowedMu.Unlock()

		var sb strings.Builder
		sb.WriteString("現在の許可パス:\n")
		if len(ap.Dirs) == 0 && len(ap.Files) == 0 {
			sb.WriteString("（まだ何も登録されていません）\n")
		}
		for _, d := range ap.Dirs {
			sb.WriteString(fmt.Sprintf("- ディレクトリ: %s\n", d))
		}
		for _, f := range ap.Files {
			sb.WriteString(fmt.Sprintf("- ファイル: %s\n", f))
		}
		s.ChannelMessageSend(m.ChannelID, sb.String())
		return

	case content == "!clearpaths":
		allowedMu.Lock()
		delete(allowedData, m.ChannelID)
		allowedMu.Unlock()
		saveAllowedPaths()
		s.ChannelMessageSend(m.ChannelID, "許可リストをクリアしました")
		return
	}

	if !strings.HasPrefix(content, "!agent ") {
		return
	}

	query := strings.TrimSpace(strings.TrimPrefix(content, "!agent "))
	if query == "" {
		s.ChannelMessageSend(m.ChannelID, "使い方例: !agent ビスケットのRITZについて調べて")
		return
	}

	history, err := loadHistory(m.ChannelID)
	if err != nil {
		s.ChannelMessageSend(m.ChannelID, "履歴読み込みエラー")
		return
	}

	messages := []Message{
		{Role: "system", Content: `あなたは優秀な日本語AIエージェントです。
ツールが使えます：
- search_web : インターネットの最新情報、天気、ニュース、商品など外部知識が必要なときは**必ず**これを使ってください。使わないと回答できません。
- execute_python_code / read_file / write_file / list_dir : 必要に応じて

ツールの結果が返ってきたら、それを基に**必ず**わかりやすい日本語で回答してください。
「検索できません」「わかりません」など否定せず、結果を活用して自然に答えてください。`},
	}
	messages = append(messages, history...)
	messages = append(messages, Message{Role: "user", Content: query})

	var finalAnswer string
	maxLoops := 8

	for i := 0; i < maxLoops; i++ {
		resp, err := callOllamaChat(messages, []Tool{
			pythonTool, readFileTool, listDirTool, writeFileTool, searchWebTool,
		})
		if err != nil {
			s.ChannelMessageSend(m.ChannelID, "Ollama通信エラー: "+err.Error())
			return
		}

		assistantMsg := resp.Message
		messages = append(messages, assistantMsg)

		if len(assistantMsg.ToolCalls) == 0 {
			finalAnswer = assistantMsg.Content
			break
		}

		for _, tc := range assistantMsg.ToolCalls {
			var toolResult string

			switch tc.Function.Name {
			case "execute_python_code":
				var args struct{ Code string }
				json.Unmarshal(tc.Function.Arguments, &args)
				toolResult = executePythonCode(args.Code)

			case "read_file":
				var args struct{ Path string }
				json.Unmarshal(tc.Function.Arguments, &args)
				toolResult = readFileToolFunc(args.Path, m.ChannelID)

			case "list_dir":
				var args struct{ Path string }
				json.Unmarshal(tc.Function.Arguments, &args)
				toolResult = listDirToolFunc(args.Path, m.ChannelID)

			case "write_file":
				var args struct {
					Path    string `json:"path"`
					Content string `json:"content"`
				}
				json.Unmarshal(tc.Function.Arguments, &args)
				toolResult = writeFileToolFunc(args.Path, args.Content, m.ChannelID)

			case "search_web":
				var args struct {
					Query      string `json:"query"`
					Engines    string `json:"engines"`
					Categories string `json:"categories"`
					Lang       string `json:"lang"`
					Safesearch int    `json:"safesearch"`
				}
				if err := json.Unmarshal(tc.Function.Arguments, &args); err != nil {
					toolResult = "引数パースエラー: " + err.Error()
					break
				}

				if args.Lang == "" {
					args.Lang = "ja"
				}
				if args.Safesearch == 0 {
					args.Safesearch = 1
				}

				searchResult, err := searchViaSearxNG(
					args.Query,
					args.Engines,
					args.Categories,
					args.Lang,
					args.Safesearch,
				)
				if err != nil {
					searchResult = "検索エラー: " + err.Error()
				}
				toolResult = searchResult

			default:
				toolResult = "未対応のツールです"
			}

			messages = append(messages, Message{
				Role:    "tool",
				Content: toolResult,
			})
		}
	}

	// ★ここが重要★ ツール結果が最新なら強制的に最終回答を生成
	if len(messages) > 0 && messages[len(messages)-1].Role == "tool" {
		messages = append(messages, Message{
			Role:    "user",
			Content: "上記のツール結果を基に、ユーザーの質問に自然な日本語で丁寧にまとめて回答してください。検索結果があれば要約してURLも適宜含めて。否定せず必ず答えて。",
		})

		finalResp, err := callOllamaChat(messages, []Tool{
			pythonTool, readFileTool, listDirTool, writeFileTool, searchWebTool,
		})
		if err == nil {
			if len(finalResp.Message.ToolCalls) == 0 {
				finalAnswer = finalResp.Message.Content
			} else {
				finalAnswer = "モデルがまだツールを使おうとしています... " + finalResp.Message.Content
			}
		} else {
			finalAnswer = "最終回答生成エラー: " + err.Error()
		}
	}

	if finalAnswer == "" {
		finalAnswer = "回答を生成できませんでした。ツール処理が複雑すぎる可能性があります。\nもう一度質問を言い換えてみてください。"
	}

	display := finalAnswer
	if len(display) > 1900 {
		display = display[:1900] + "\n…（続きはログ参照）"
	}

	s.ChannelMessageSend(m.ChannelID, display)

	logPath := saveLog(query, finalAnswer)
	if logPath != "" {
		s.ChannelMessageSend(m.ChannelID, "ログ保存: "+logPath)
	}

	history = append(history, Message{Role: "user", Content: query})
	history = append(history, Message{Role: "assistant", Content: finalAnswer})
	saveHistory(m.ChannelID, history)
}

func main() {
	loadAllowedPaths()

	token := os.Getenv("DISCORD_TOKEN")
	if token == "" {
		fmt.Println("DISCORD_TOKEN が設定されていません")
		os.Exit(1)
	}

	dg, err := discordgo.New("Bot " + token)
	if err != nil {
		fmt.Printf("Discordセッションエラー: %v\n", err)
		os.Exit(1)
	}

	dg.AddHandler(messageHandler)

	dg.Identify.Intents = discordgo.IntentsGuildMessages | discordgo.IntentsMessageContent

	err = dg.Open()
	if err != nil {
		fmt.Printf("接続エラー: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Bot起動中... !agent で質問してください")

	sc := make(chan os.Signal, 1)
	signal.Notify(sc, syscall.SIGINT, syscall.SIGTERM)
	<-sc

	dg.Close()
}
