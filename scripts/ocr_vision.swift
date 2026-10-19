// macOS Vision OCR — 표준입력으로 이미지 경로를 한 줄씩 받아, 위쪽 띠(범례 줄)의 글자를 JSON 한 줄로 낸다.
// 사용: printf '%s\n' a.png b.png | swift scripts/ocr_vision.swift
// 출력: {"path": ..., "obs": [{"t": 글자, "x": 왼쪽(0~1), "y": 세로 가운데(0~1, 아래가 0), "h": 높이}]}
import Foundation
import ImageIO
import Vision

func recognize(_ path: String) -> [[String: Any]] {
    let url = URL(fileURLWithPath: path)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return [] }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.regionOfInterest = CGRect(x: 0, y: 0.84, width: 0.62, height: 0.16)  // 범례가 있는 왼쪽 위
    let handler = VNImageRequestHandler(cgImage: image, options: [:])
    try? handler.perform([request])
    var out: [[String: Any]] = []
    for obs in request.results ?? [] {
        guard let text = obs.topCandidates(1).first?.string else { continue }
        // regionOfInterest 안의 좌표 → 전체 이미지 좌표
        let box = obs.boundingBox
        let x = 0.0 + box.minX * 0.62
        let y = 0.84 + box.midY * 0.16
        out.append(["t": text, "x": x, "y": y, "h": box.height * 0.16])
    }
    return out
}

while let line = readLine() {
    let path = line.trimmingCharacters(in: .whitespacesAndNewlines)
    if path.isEmpty { continue }
    let obj: [String: Any] = ["path": path, "obs": recognize(path)]
    if let data = try? JSONSerialization.data(withJSONObject: obj),
       let text = String(data: data, encoding: .utf8) {
        print(text)
        fflush(stdout)
    }
}
