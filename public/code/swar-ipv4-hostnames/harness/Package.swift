// swift-tools-version: 6.3
import PackageDescription
let package = Package(
  name: "H",
  dependencies: [.package(path: "../src/swift-endpoint", traits: [])],
  targets: [.executableTarget(name: "H", dependencies: [.product(name: "IPAddress", package: "swift-endpoint")],
    swiftSettings: [.unsafeFlags(["-Ounchecked"])])]
)
