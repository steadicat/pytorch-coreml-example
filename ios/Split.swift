//
//  SplitSequence.swift
//  Movement
//
//  Created by Stefano J. Attardi on 1/19/18.
//  Copyright © 2018 Rational Creation. All rights reserved.
//

import CoreML

@objc(Split)
class Split: NSObject {

  var model: AnyObject? = nil
  
  @objc static func requiresMainQueueSetup() -> Bool {
    return false
  }

  @objc(split:callback:)
  func split(points: [[Float32]], callback: RCTResponseSenderBlock) {
    guard points.count >= 2 else {
      callback([NSNull(), []])
      return
    }
    if #available(iOS 11.0, *) {
      if self.model == nil {
        self.model = SplitModel()
      }
      guard let model = self.model as? SplitModel else {
        print("Failed to create model")
        callback(["coreml_error", NSNull()])
        return
      }

      // let example: [[Float32]] = [[41, 24], [163, 116], [254, 116], [319, 103], [484, 112], [533, 84], [629, 91]]
      let xs = points.map { $0[0] }
      let ys = points.map { $0[1] }
      let minX = xs.min()!
      let maxX = xs.max()!
      let minY = ys.min()!
      let maxY = ys.max()!
      let yShift = ((maxY - minY) / (maxX - minX)) / 2.0
      guard let data = try? MLMultiArray(shape: [1, 2, 100], dataType: .float32) else {
        print("Failed to create MLMultiArray")
        callback(["coreml_error", NSNull()])
        return
      }
      
      for (i, point) in points.enumerated() {
        let doubleI = Double(i)
        let x = Double((point[0] - minX) / (maxX - minX) - 0.5)
        let y = Double((point[1] - minY) / (maxX - minX) - yShift)
        data[[NSNumber(floatLiteral: 0), NSNumber(floatLiteral: 0), NSNumber(floatLiteral: doubleI)]] = NSNumber(floatLiteral: x)
        data[[NSNumber(floatLiteral: 0), NSNumber(floatLiteral: 1), NSNumber(floatLiteral: doubleI)]] = NSNumber(floatLiteral: y)
      }
            
      do {
        let start = CACurrentMediaTime()
        let prediction = try model.prediction(_1: data)._27
        print("ml time \(CACurrentMediaTime() - start)")
        var indices: [Int] = []
        for (index, prob) in prediction {
          if prob > 0.5 && index < points.count - 1 {
            indices.append(Int(index))
          }
        }
        callback([NSNull(), indices.sorted()])
        return
      } catch {
        print("Error running CoreML: \(error)")
        callback(["coreml_error", NSNull()])
        return
      }
    } else {
      callback(["coreml_unavailable", NSNull()])
    }
  }
}

