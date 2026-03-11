"""exceptions.py 单元测试 — 异常层级验证"""

import pytest
from robotmem.exceptions import (
    DatabaseError,
    EmbeddingError,
    RobotMemError,
    ValidationError,
)


class TestExceptionHierarchy:
    """异常继承关系"""

    def test_validation_error_is_robotmem_error(self):
        assert issubclass(ValidationError, RobotMemError)

    def test_database_error_is_robotmem_error(self):
        assert issubclass(DatabaseError, RobotMemError)

    def test_embedding_error_is_robotmem_error(self):
        assert issubclass(EmbeddingError, RobotMemError)

    def test_robotmem_error_is_exception(self):
        assert issubclass(RobotMemError, Exception)

    def test_catch_all_with_base_class(self):
        """用户可以 except RobotMemError 捕获所有 robotmem 异常"""
        for exc_cls in (ValidationError, DatabaseError, EmbeddingError):
            with pytest.raises(RobotMemError):
                raise exc_cls("test")

    def test_catch_specific_not_others(self):
        """各异常类型互不捕获"""
        with pytest.raises(ValidationError):
            raise ValidationError("bad param")
        # DatabaseError 不应被 ValidationError 捕获
        with pytest.raises(DatabaseError):
            raise DatabaseError("lock timeout")

    def test_message_preserved(self):
        err = ValidationError("insight 不能为空")
        assert str(err) == "insight 不能为空"

    def test_exception_chaining(self):
        """支持 raise ... from e 异常链"""
        original = ValueError("原始错误")
        try:
            raise DatabaseError("写入失败") from original
        except DatabaseError as e:
            assert e.__cause__ is original
