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
	Role    string        `json:"role"`
	Content []ContentPart `json:"content"`
}

type ChatRequest struct {
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream"`
	Temperature float64   `json:"temperature,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
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
}

// imageToDataURI reads an image file and returns a base64 data URI
func imageToDataURI(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read image %s: %w", path, err)
	}

	ext := strings.ToLower(filepath.Ext(path))
	mimeTypes := map[string]string{
		".png":  "image/png",
		".jpg":  "image/jpeg",
		".jpeg": "image/jpeg",
		".gif":  "image/gif",
		".webp": "image/webp",
		".bmp":  "image/bmp",
	}

	mime, ok := mimeTypes[ext]
	if !ok {
		mime = "image/png" // fallback
	}

	encoded := base64.StdEncoding.EncodeToString(data)
	return fmt.Sprintf("data:%s;base64,%s", mime, encoded), nil
}

func main() {
	host := flag.String("host", "127.0.0.1", "llama-server host")
	port := flag.Int("port", 8080, "llama-server port")
	promptFile := flag.String("prompt-file", "", "path to prompt text file")
	prompt := flag.String("prompt", "", "prompt text (alternative to --prompt-file)")
	stream := flag.Bool("stream", false, "enable streaming output")
	temp := flag.Float64("temp", 0.7, "sampling temperature")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] --image img1.png [--image img2.png ...]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "A client for llama-server with vision/multimodal support.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.VisitAll(func(f *flag.Flag) {
			fmt.Fprintf(os.Stderr, "  --%-12s %s", f.Name, f.Usage)
			if f.DefValue != "" {
				fmt.Fprintf(os.Stderr, " (default %s)", f.DefValue)
			}
			fmt.Fprintln(os.Stderr)
		})
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s --host 10.0.0.1 --port 8080 --image photo.jpg --prompt 'Describe this image'\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --image a.png --image b.png --prompt-file prompt.txt --stream\n", os.Args[0])
	}

	// collect --image flags (multiple allowed)
	var images []string
	for i, arg := range os.Args {
		if arg == "--image" || arg == "-image" {
			if i+1 < len(os.Args) {
				images = append(images, os.Args[i+1])
			}
		}
	}

	// remove --image and its value from os.Args before flag.Parse
	var filteredArgs []string
	filteredArgs = append(filteredArgs, os.Args[0])
	skip := false
	for i := 1; i < len(os.Args); i++ {
		if skip {
			skip = false
			continue
		}
		if os.Args[i] == "--image" || os.Args[i] == "-image" {
			skip = true
			continue
		}
		filteredArgs = append(filteredArgs, os.Args[i])
	}
	os.Args = filteredArgs

	flag.Parse()

	// resolve prompt text
	var promptText string
	if *promptFile != "" {
		data, err := os.ReadFile(*promptFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading prompt file: %v\n", err)
			os.Exit(1)
		}
		promptText = string(data)
	} else if *prompt != "" {
		promptText = *prompt
	} else {
		fmt.Fprintf(os.Stderr, "Error: either --prompt or --prompt-file is required\n")
		flag.Usage()
		os.Exit(1)
	}

	if len(images) == 0 {
		fmt.Fprintf(os.Stderr, "Info: no --image provided, sending text-only request\n")
	}

	// build message content parts
	var parts []ContentPart
	parts = append(parts, ContentPart{Type: "text", Text: promptText})

	for _, imgPath := range images {
		dataURI, err := imageToDataURI(imgPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		parts = append(parts, ContentPart{
			Type:     "image_url",
			ImageURL: &ImageURL{URL: dataURI},
		})
	}

	req := ChatRequest{
		Messages:    []Message{{Role: "user", Content: parts}},
		Stream:      *stream,
		Temperature: *temp,
	}

	body, err := json.Marshal(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error marshaling request: %v\n", err)
		os.Exit(1)
	}

	url := fmt.Sprintf("http://%s:%d/v1/chat/completions", *host, *port)

	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(body))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
		os.Exit(1)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	tStart := time.Now()

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error connecting to %s: %v\n", url, err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		fmt.Fprintf(os.Stderr, "Server error (HTTP %d): %s\n", resp.StatusCode, string(errBody))
		os.Exit(1)
	}

	if *stream {
		// SSE stream parsing
		scanner := bufio.NewScanner(resp.Body)
		var ttft time.Duration
		firstToken := true
		tokenCount := 0
		var lastUsage *Usage

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
			for _, c := range chunk.Choices {
				if c.Delta.Content != "" {
					if firstToken {
						ttft = time.Since(tStart)
						firstToken = false
					}
					tokenCount++
					fmt.Print(c.Delta.Content)
				}
			}
		}
		fmt.Println()

		tTotal := time.Since(tStart)

		// print stats
		fmt.Fprintf(os.Stderr, "\n--- stats ---\n")
		if tokenCount > 0 {
			fmt.Fprintf(os.Stderr, "TTFT:          %s\n", ttft.Round(time.Millisecond))
		}
		fmt.Fprintf(os.Stderr, "Total time:    %s\n", tTotal.Round(time.Millisecond))
		if tokenCount > 1 {
			genTime := tTotal - ttft
			if genTime > 0 {
				tps := float64(tokenCount-1) / genTime.Seconds()
				fmt.Fprintf(os.Stderr, "Tokens:        %d\n", tokenCount)
				fmt.Fprintf(os.Stderr, "Speed:         %.2f tok/s\n", tps)
			}
		} else if tokenCount == 1 {
			fmt.Fprintf(os.Stderr, "Tokens:        1\n")
		}
		if lastUsage != nil {
			fmt.Fprintf(os.Stderr, "Prompt tokens: %d\n", lastUsage.PromptTokens)
			fmt.Fprintf(os.Stderr, "Compl tokens:  %d\n", lastUsage.CompletionTokens)
			fmt.Fprintf(os.Stderr, "Total tokens:  %d\n", lastUsage.TotalTokens)
		}
	} else {
		// non-streaming: read raw body so we can measure timing
		rawBody, err := io.ReadAll(resp.Body)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading response: %v\n", err)
			os.Exit(1)
		}
		ttft := time.Since(tStart) // for non-stream, TTFT ≈ total time
		tTotal := ttft

		var result ChatResponse
		if err := json.Unmarshal(rawBody, &result); err != nil {
			fmt.Fprintf(os.Stderr, "Error decoding response: %v\n", err)
			os.Exit(1)
		}
		if len(result.Choices) > 0 {
			fmt.Println(result.Choices[0].Message.Content)
		}

		// print stats
		fmt.Fprintf(os.Stderr, "\n--- stats ---\n")
		fmt.Fprintf(os.Stderr, "TTFT:          %s\n", ttft.Round(time.Millisecond))
		fmt.Fprintf(os.Stderr, "Total time:    %s\n", tTotal.Round(time.Millisecond))
		if result.Usage != nil {
			fmt.Fprintf(os.Stderr, "Prompt tokens: %d\n", result.Usage.PromptTokens)
			fmt.Fprintf(os.Stderr, "Compl tokens:  %d\n", result.Usage.CompletionTokens)
			fmt.Fprintf(os.Stderr, "Total tokens:  %d\n", result.Usage.TotalTokens)
			if tTotal.Seconds() > 0 {
				tps := float64(result.Usage.CompletionTokens) / tTotal.Seconds()
				fmt.Fprintf(os.Stderr, "Speed:         %.2f tok/s\n", tps)
			}
		}
	}
}
