# 消融实验3旧候选编译停止记录

- 会话：`ablation3-gpt67-tp8-scope-20261016-221018-retry1`
- 状态：候选编译已人工中止，未进入 mapping、预检或正式性能测试。
- 原因：编译进程继承了 `DEBUG=release`，NCCL 源码构建日志出现 `-O0 -G -g`；该构建不作为正式实验输入。
- 现场：`outputs/h100/phase6/ablation3-gpt67-tp8-scope-20261016-221018-retry1/candidates/` 及各 `compile.log` 保留，不覆盖、不宣称 built。
- 修正：新会话使用已核验的历史开发库 `outputs/h100/phase6/tp8-20261009-deployment/nccl-release`，显式 `DEBUG=0`，`MAKEFLAGS` 为空，CC/CXX/CUDAHOSTCXX 均固定为 GCC/G++ 11。
- 历史 NCCL 哈希：`include/nccl.h` = `af992d0e111074a978268e9f433b943409a1aba8e6dcd322107bef185eb0010d`；`lib/libnccl_static.a` = `ddb42b57da73726deb1e76871e73207577dbad77323dff523130d17d2da7da82`。
