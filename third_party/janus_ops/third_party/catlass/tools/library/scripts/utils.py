#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os


class KernelGroupFile:
    def __init__(self, file_name):
        self.file_name = file_name
        self.operation_headers = set()
        self.kernel_instance_headers = set()
        self.custom_common_decls = set()
        self.body_src = []

        self.header_template = """
{operation_headers}

{kernel_instance_headers}

namespace Catlass {{
namespace Library {{
using namespace Catlass;

{custom_common_decls}
"""

        self.tail = """
}
}
"""

    def add_headers(self, headers):
        self.operation_headers.add(headers)

    def add_instance(self, custom_header, custom_common_decls, body):
        self.kernel_instance_headers.add(custom_header)
        self.custom_common_decls.add(custom_common_decls)
        self.body_src.append(body)

    def write_in_dir(self, workspace_dir):
        operation_headers = ''
        for header in self.operation_headers:
            operation_headers += header + '\n'
        kernel_instance_headers = ''
        for header in self.kernel_instance_headers:
            kernel_instance_headers += header + '\n'
        custom_common_decls_src = ''
        for decl in self.custom_common_decls:
            custom_common_decls_src += decl + '\n'
        headers = self.header_template.format(
            operation_headers=operation_headers,
            kernel_instance_headers=kernel_instance_headers,
            custom_common_decls=custom_common_decls_src
        )

        path = os.path.join(workspace_dir, self.file_name)
        with open(path, "w") as f:
            f.write(headers)
            for body in self.body_src:
                f.write(body)
            f.write(self.tail)
