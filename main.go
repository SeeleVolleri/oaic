package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// stringSlice implements flag.Value for repeatable flags.
type stringSlice []string

func (s *stringSlice) String() string { return strings.Join(*s, ", ") }
func (s *stringSlice) Set(v string) error {
	*s = append(*s, v)
	return nil
}

// OpenAI-compatible request/response types

type ImageURL struct {
	URL string `json:"url"`
}

type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type Message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type ChatRequest struct {
	Model         string         `json:"model,omitempty"`
	Messages      []Message      `json:"messages"`
	Stream        bool           `json:"stream"`
	StreamOptions *StreamOptions `json:"stream_options,omitempty"`
	Temperature   *float64       `json:"temperature,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type Timings struct {
	PromptN            int     `json:"prompt_n"`
	PromptMS           float64 `json:"prompt_ms"`
	PromptPerSecond    float64 `json:"prompt_per_second"`
	PredictedN         int     `json:"predicted_n"`
	PredictedMS        float64 `json:"predicted_ms"`
	PredictedPerSecond float64 `json:"predicted_per_second"`
}

type Choice struct {
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
}

type ChatResponse struct {
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
	Timings *Timings `json:"timings,omitempty"`
}

type StreamDelta struct {
	Content string `json:"content"`
}

type StreamChoice struct {
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

type StreamChunk struct {
	Choices []StreamChoice `json:"choices"`
	Usage   *Usage         `json:"usage,omitempty"`
	Timings *Timings       `json:"timings,omitempty"`
}

// mimeByExt maps file extensions to MIME types.
var mimeByExt = map[string]string{
	".png":  "image/png",
	".jpg":  "image/jpeg",
	".jpeg": "image/jpeg",
	".gif":  "image/gif",
	".webp": "image/webp",
	".bmp":  "image/bmp",
}

// imageToDataURI reads an image file and returns a base64 data URI.
func imageToDataURI(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read image %s: %w", path, err)
	}

	mime := mimeByExt[strings.ToLower(filepath.Ext(path))]
	if mime == "" {
		mime = "application/octet-stream"
	}

	encoded := base64.StdEncoding.EncodeToString(data)
	return fmt.Sprintf("data:%s;base64,%s", mime, encoded), nil
}

func fatal(format string, a ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", a...)
	os.Exit(1)
}

func main() {
	var images stringSlice

	baseURL := flag.String("url", "http://127.0.0.1:8080", "base URL of the API server")
	model := flag.String("model", "", "model name (e.g. gpt-4o, llama3)")
	token := flag.String("token", "", "bearer token (default: OPENAI_API_KEY env)")
	promptFile := flag.String("prompt-file", "", "path to prompt text file (use - for stdin)")
	prompt := flag.String("prompt", "", "prompt text (alternative to --prompt-file)")
	systemPrompt := flag.String("system-prompt", "", "system prompt text")
	stream := flag.Bool("stream", false, "enable streaming output")
	verbose := flag.Bool("verbose", false, "print timing and token stats to stderr")
	temp := flag.Float64("temp", -1, "sampling temperature (default: not sent)")
	timeout := flag.Int("timeout", 300, "HTTP timeout in seconds (0 = no timeout)")
	flag.Var(&images, "image", "image file path (repeatable)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] [--image FILE ...]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Client for OpenAI-compatible chat completion APIs with vision support.\n")
		fmt.Fprintf(os.Stderr, "Prompt is read from --prompt, --prompt-file, or stdin.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.VisitAll(func(f *flag.Flag) {
			fmt.Fprintf(os.Stderr, "  --%-12s %s", f.Name, f.Usage)
			if f.DefValue != "" && f.DefValue != "false" && f.DefValue != "[]" && f.DefValue != "-1" {
				fmt.Fprintf(os.Stderr, " (default %s)", f.DefValue)
			}
			fmt.Fprintln(os.Stderr)
		})
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s --image photo.jpg --prompt 'Describe this image'\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  echo 'Describe this' | %s --image photo.jpg\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --url https://api.example.com --model gpt-4o --image a.png --prompt-file prompt.txt --stream\n", os.Args[0])
	}

	flag.Parse()

	// resolve prompt text: --prompt > --prompt-file > stdin
	var promptText string
	switch {
	case *prompt != "":
		promptText = *prompt
	case *promptFile == "-":
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			fatal("reading stdin: %v", err)
		}
		promptText = string(data)
	case *promptFile != "":
		data, err := os.ReadFile(*promptFile)
		if err != nil {
			fatal("reading prompt file: %v", err)
		}
		promptText = string(data)
	default:
		fi, _ := os.Stdin.Stat()
		if fi != nil && fi.Mode()&os.ModeCharDevice == 0 {
			data, err := io.ReadAll(os.Stdin)
			if err != nil {
				fatal("reading stdin: %v", err)
			}
			promptText = string(data)
		}
	}

	promptText = strings.TrimSpace(promptText)
	if promptText == "" {
		fatal("no prompt provided (use --prompt, --prompt-file, or pipe to stdin)")
	}

	// build message content parts
	var parts []ContentPart
	parts = append(parts, ContentPart{Type: "text", Text: promptText})

	for _, imgPath := range images {
		dataURI, err := imageToDataURI(imgPath)
		if err != nil {
			fatal("%v", err)
		}
		parts = append(parts, ContentPart{
			Type:     "image_url",
			ImageURL: &ImageURL{URL: dataURI},
		})
	}

	userContent, err := json.Marshal(parts)
	if err != nil {
		fatal("marshaling content: %v", err)
	}

	var messages []Message
	if *systemPrompt != "" {
		sysContent, _ := json.Marshal(*systemPrompt)
		messages = append(messages, Message{Role: "system", Content: sysContent})
	}
	messages = append(messages, Message{Role: "user", Content: userContent})

	req := ChatRequest{
		Model:    *model,
		Messages: messages,
		Stream:   *stream,
	}
	if *stream && *verbose {
		req.StreamOptions = &StreamOptions{IncludeUsage: true}
	}
	if *temp >= 0 {
		req.Temperature = temp
	}

	body, err := json.Marshal(req)
	if err != nil {
		fatal("marshaling request: %v", err)
	}

	url := strings.TrimRight(*baseURL, "/") + "/v1/chat/completions"

	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		fatal("creating request: %v", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	authToken := *token
	if authToken == "" {
		authToken = os.Getenv("OPENAI_API_KEY")
	}
	if authToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+authToken)
	}

	client := &http.Client{}
	if *timeout > 0 {
		client.Timeout = time.Duration(*timeout) * time.Second
	}

	tStart := time.Now()

	resp, err := client.Do(httpReq)
	if err != nil {
		fatal("%v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		fmt.Fprintf(os.Stderr, "%s\n", string(errBody))
		os.Exit(1)
	}

	if *stream {
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 1024*1024), 1024*1024)

		var ttft time.Duration
		firstToken := true
		var lastUsage *Usage
		var lastTimings *Timings

		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}
			var chunk StreamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}
			if chunk.Usage != nil {
				lastUsage = chunk.Usage
			}
			if chunk.Timings != nil {
				lastTimings = chunk.Timings
			}
			for _, c := range chunk.Choices {
				if c.Delta.Content != "" {
					if firstToken {
						ttft = time.Since(tStart)
						firstToken = false
					}
					fmt.Print(c.Delta.Content)
				}
			}
		}
		if err := scanner.Err(); err != nil {
			fatal("reading stream: %v", err)
		}
		fmt.Println()

		if *verbose {
			tTotal := time.Since(tStart)

			fmt.Fprintf(os.Stderr, "TTFT:          %s\n", ttft.Round(time.Millisecond))
			fmt.Fprintf(os.Stderr, "Total time:    %s\n", tTotal.Round(time.Millisecond))
			if lastUsage != nil {
				fmt.Fprintf(os.Stderr, "Prompt tokens: %d\n", lastUsage.PromptTokens)
				fmt.Fprintf(os.Stderr, "Compl tokens:  %d\n", lastUsage.CompletionTokens)
				fmt.Fprintf(os.Stderr, "Total tokens:  %d\n", lastUsage.TotalTokens)
			}
			if lastTimings != nil && lastTimings.PredictedPerSecond > 0 {
				fmt.Fprintf(os.Stderr, "Speed:         %.2f tok/s (server)\n", lastTimings.PredictedPerSecond)
			} else {
				genTime := tTotal - ttft
				if genTime > 0 && lastUsage != nil && lastUsage.CompletionTokens > 0 {
					tps := float64(lastUsage.CompletionTokens) / genTime.Seconds()
					fmt.Fprintf(os.Stderr, "Speed:         %.2f tok/s (client)\n", tps)
				}
			}
		}
	} else {
		rawBody, err := io.ReadAll(resp.Body)
		if err != nil {
			fatal("reading response: %v", err)
		}
		tTotal := time.Since(tStart)

		var result ChatResponse
		if err := json.Unmarshal(rawBody, &result); err != nil {
			fatal("decoding response: %v", err)
		}
		if len(result.Choices) == 0 {
			fatal("server returned no choices")
		}
		fmt.Println(result.Choices[0].Message.Content)

		if *verbose {
			fmt.Fprintf(os.Stderr, "Latency:       %s\n", tTotal.Round(time.Millisecond))
			if result.Usage != nil {
				fmt.Fprintf(os.Stderr, "Prompt tokens: %d\n", result.Usage.PromptTokens)
				fmt.Fprintf(os.Stderr, "Compl tokens:  %d\n", result.Usage.CompletionTokens)
				fmt.Fprintf(os.Stderr, "Total tokens:  %d\n", result.Usage.TotalTokens)
			}
			if result.Timings != nil && result.Timings.PredictedPerSecond > 0 {
				fmt.Fprintf(os.Stderr, "Speed:         %.2f tok/s (server)\n", result.Timings.PredictedPerSecond)
			} else if result.Usage != nil && tTotal.Seconds() > 0 {
				tps := float64(result.Usage.CompletionTokens) / tTotal.Seconds()
				fmt.Fprintf(os.Stderr, "Speed:         %.2f tok/s (client)\n", tps)
			}
		}
	}
}
