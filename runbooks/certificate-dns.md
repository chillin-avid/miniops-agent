# 证书与 DNS 故障

## 证书问题
- `certificate expired` 表示证书过期；hostname mismatch 表示访问域名不在证书范围内。
- 检查证书有效期、完整证书链、访问域名和客户端时间。
- 更换证书后要确认负载均衡和所有实例都已加载新版本。

## DNS 问题
- `name resolution failed`、`NXDOMAIN` 或解析到旧地址通常与 DNS 有关。
- 从故障实例执行解析，检查记录、TTL、本地缓存和上游 DNS。
- 不要长期用 hosts 文件绕过 DNS，它只能作为短期验证手段。

## 处理原则
- 证书和 DNS 修改都需要验证多个网络位置，避免只在本机判断恢复。

